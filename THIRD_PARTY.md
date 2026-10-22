# Model provenance

| File | Source | Pinned source commit | License |
| --- | --- | --- | --- |
| `models/fastestdet.{param,bin}` | [FastestDet](https://github.com/dog-qiuqiu/FastestDet), `example/ncnn` | `50473cd155cb088aa4a99e64ff6a4b3c24fa07e1` | BSD-3-Clause; `models/FASTESTDET_LICENSE.md` |
| `models/trilitenet-tiny-320.{param,bin}` | [TriLiteNet](https://github.com/chequanghuy/TriLiteNet), Tiny checkpoint | `4ac49477ff73dc9f45497ad4c397e92e6089436b` | MIT; `models/TRILITENET_LICENSE` |

The optional TriLiteNet graph was converted from the original PyTorch checkpoint at 320×192 with `scripts/export_trilitenet.py` and pnnx, with a mathematically equivalent fixed-bin attention rewrite checked against the original PyTorch outputs. Only the `out4` lane head is extracted in production. This conversion is experimental and its lane predictions require road-specific evaluation.

Both model hashes, preprocessing parameters, output names, and source locations are recorded in `models/manifest.json`. Runtime startup checks both model file hashes before loading. `detector.py` implements the FastestDet NCNN decoder with vehicle-only COCO class filtering and class-aware non-maximum suppression. The weights are included so a Pi install can run without an export toolchain or a model download.
