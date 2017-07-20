from scipy import stats
import scipy
import numpy as np
import time
import math

def getMode(case):
    counts = [0]*5
    for vote in case:
        counts[vote-1]+=1
    return round(getMax(counts))+1

def getMax(lst, verbose=False):
    mx = max(lst)
    mx_vals = []
    
    for k,x in enumerate(lst):
        if x == mx:
            mx_vals.append(k)
    print(mx_vals)
    if len(mx_vals) == 1:
        return mx_vals[0]
    else:
        return (sum(mx_vals)/len(mx_vals)) 

mode = int(scipy.stats.mode(np.array([2,3,4,5]))[0][0])

print("[3,2,1,4]: ", mode)

mode = getMode([2,3,4,5])

print("[2,3,4,5]: ", mode)



