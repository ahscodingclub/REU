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

def getMode(case):
    counts = [0]*5
    for vote in case:
        counts[vote-1]+=1
    return getMax(counts)+1

print(getMode([2,2,3,4]))
print(getMode([2,3,4,5]))
