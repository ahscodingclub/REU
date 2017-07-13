def median(case):
    case.sort()
    while(len(case) > 2):
        case = case[1:-1]
        if len(case) == 1:
            return case[0]
        else:
            return (float(case[0])+case[1])/2


print(median([2,2,3,4]))


