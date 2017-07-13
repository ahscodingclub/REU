from scipy import stats
import scipy
import numpy as np
import time
import math

def getMedian(case):
    case.sort()
    while(len(case) > 2):
        case = case[1:-1]
        if len(case) == 1:
            return case[0]
        else:
            return (float(case[0])+case[1])/2

start = time.time()
for i in range(0,50000):
  pign = [0]*5
  a = np.array([2,2,3,4])
  median = np.median(a)
  if float(median).is_integer():
    pign[int(math.floor(median))-1] = 1
  else:
    pign[int(math.floor(median))-1] = 1 - (median - math.floor(median))
    if median-math.floor(median) != 0:
      pign[int(math.floor(median))] = median - math.floor(median)

stop = time.time()

print("process spent: ", float(stop-start))

start = time.time()
for i in range(0,50000):
  pign = [0]*5
  median = getMedian([2,2,3,4])
  if float(median).is_integer():
    pign[int(math.floor(median))-1] = 1
  else:
    pign[int(math.floor(median))-1] = 1 - (median - math.floor(median))
    if median-math.floor(median) != 0:
      pign[int(math.floor(median))] = median - math.floor(median)
stop = time.time()

print("process spent: ", float(stop-start))
