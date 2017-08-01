def getMean(lst):
    sum = 0
    for i in lst:     # Count number of instances of each rating
        sum += i
        mean = sum/4
    return mean

def getMedian(case):
    case.sort()
    while(len(case) > 2):
        case = case[1:-1]
        if len(case) == 1:
            return case[0]
        else:
            return round((float(case[0])+case[1])/2)

def getMode(case):
    counts = [0]*5
    for vote in case:
        counts[int(vote)-1]+=1
    return int(round(getMax(counts))+1)

def getMax(lst):
    mx = max(lst)
    mx_vals = []

    for k,x in enumerate(lst):
        if x == mx:
            mx_vals.append(k)
    if len(mx_vals) == 1:
        return mx_vals[0]
    else:
        return (sum(mx_vals)/len(mx_vals))
