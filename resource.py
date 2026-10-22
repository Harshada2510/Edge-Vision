
RUSAGE_SELF = 0

class Usage:
    ru_maxrss = 0
    ru_utime = 0
    ru_stime = 0

def getrusage(who):
    return Usage()
