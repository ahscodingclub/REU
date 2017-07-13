from scipy import stats
import scipy
import numpy as np
import time
import math

def getMode(case):
    counts = [0]*5
    for vote in case:
        counts[vote-1]+=1
    return getMax(counts)+1

def getMax(lst, verbose=False):
    mx = max(lst)
    mx_vals = []
    
    for k,x in enumerate(lst):
        if x == mx:
            mx_vals.append(k)
    if len(mx_vals) == 1:
        return mx_vals[0]
    else:
        return (sum(mx_vals)/len(mx_vals))

start = time.time()
for i in range(0,50000):
    mode = int(scipy.stats.mode([2,2,3,4])[0][0])
stop = time.time()

print("process spent: ", float(stop-start))

start = time.time()
for i in range(0,50000):
    mode = int(scipy.stats.mode(np.array([2,2,3,4]))[0][0])
stop = time.time()

print("process spent: ", float(stop-start))

start = time.time()
for i in range(0,50000):
  pign = [0]*5
  mode = getMode([2,2,3,4]) 
  pign[mode-1] = 1
stop = time.time()

print("process spent: ", float(stop-start))



