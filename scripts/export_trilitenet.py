#!/usr/bin/env python3
"""Export real TriLiteNet checkpoint with raw detection heads to fixed-shape NCNN.

Common modules are loaded directly to avoid training-only dependencies. State
dictionary loading is strict. Decode raw heads externally using checkpoint
anchor_grid, and strides 8,16,32. This is an experimental export, not upstream.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import types

import torch
import torch.nn as nn
import pnnx


def ncnn_attention(self,x):
    """Equivalent fixed-bin attention without rank-five tensor permutations."""
    b,c,h,w=x.shape
    bh,bw=self.bin_size
    rh,rw=h//bh,w//bw
    def split(t):
        return torch.stack([t[:,:,i*rh:(i+1)*rh,j*rw:(j+1)*rw].flatten(2).transpose(1,2)
                            for i in range(bh) for j in range(bw)],dim=1)
    cam=self.conv_cam(x)
    cls=self.sigmoid(self.pool_cam(cam)).flatten(2).transpose(1,2).unsqueeze(3)
    xs,cs=split(x),split(cam)
    local=torch.matmul(torch.softmax(cs,dim=2).transpose(2,3),xs)*cls
    local=self.gcn(local)
    glob=self.relu(self.fuse(local)).repeat(1,bh*bw,1,1)
    query,key,value=self.proj_query(xs),self.proj_key(local),self.proj_value(glob)
    affinity=torch.softmax(torch.matmul(query,key.transpose(2,3)),dim=-1)
    out=torch.matmul(affinity,value)
    blocks=[out[:,i].transpose(1,2).reshape(b,value.shape[-1],rh,rw) for i in range(bh*bw)]
    recovered=torch.cat([torch.cat(blocks[i*bw:(i+1)*bw],dim=3) for i in range(bh)],dim=2)
    return x+self.conv_out(recovered)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',required=True)
    ap.add_argument('--weights',required=True)
    ap.add_argument('--variant',choices=['tiny','small','base'],default='tiny')
    ap.add_argument('--height',type=int,default=384)
    ap.add_argument('--width',type=int,default=640)
    ap.add_argument('--heads',choices=['all','det-lane','det'],default='all')
    ap.add_argument('--safe-attention',action='store_true')
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('tri_common',Path(a.repo)/'lib/models/common.py')
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    cfg=common.sc_ch_dict[a.variant]
    class ExportModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model=nn.ModuleList([
                common.Encoder(cfg),common.DetectHead(cfg),
                common.Detect(1,[[4,12,7,19,11,28],[17,40,25,58,38,89],[62,136,88,206,124,412]], [cfg['chanels'][3]]*3),
                common.SegmentHead(cfg),common.UpSimpleBlock(cfg['chanels'][0],2),
                common.UpSimpleBlock(cfg['chanels'][0],2)])
        def forward(self,x):
            enc=self.model[0](x)
            features=self.model[1](enc)
            raw=tuple(self.model[2].m[i](f) for i,f in enumerate(features))
            if a.heads=='det': return raw
            da,ll=self.model[3](enc)
            if a.heads=='det-lane': return raw+(self.model[5](ll),)
            return raw+(self.model[4](da),self.model[5](ll))
    m=ExportModel().eval()
    state=torch.load(a.weights,map_location='cpu',weights_only=True)
    print(m.load_state_dict(state,strict=True),flush=True)
    for module in m.modules():
        if isinstance(module,nn.BatchNorm2d):
            # initialize_weights() in upstream sets eps=1e-3 on every BatchNorm.
            module.eps=1e-3
    torch.manual_seed(17)
    x=torch.randn(1,3,a.height,a.width)
    with torch.inference_mode():
        if a.safe_attention:
            before=m(x)
            m.model[3].caam.forward=types.MethodType(ncnn_attention,m.model[3].caam)
            after=m(x)
            errors=[float((aa-bb).abs().max()) for aa,bb in zip(before,after)]
            print('attention_rewrite_max_absolute_errors',errors,flush=True)
            for aa,bb in zip(before,after):
                torch.testing.assert_close(aa,bb,rtol=1e-5,atol=1e-5)
        out=m(x)
        print('output_shapes', [list(o.shape) for o in out],flush=True)
        path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
        torch.jit.trace(m,x).save(str(path)+'.pt')
        torch.save({'input':x,'outputs':out},str(path)+'.reference.pt')
    meta={'variant':a.variant,'heads':a.heads,'input_shape':list(x.shape),
          'output_shapes':[list(o.shape) for o in out],
          'anchor_grid':state['model.2.anchor_grid'].tolist(),
          'parameter_count':sum(p.numel() for p in m.parameters())}
    Path(str(path)+'.metadata.json').write_text(json.dumps(meta,indent=2))
    subprocess.run([pnnx.EXEC_PATH,str(path)+'.pt',f'inputshape=[1,3,{a.height},{a.width}]','fp16=0'],check=True)


if __name__=='__main__': main()
