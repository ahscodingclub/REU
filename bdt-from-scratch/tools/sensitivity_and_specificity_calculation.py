f = open('confusion_matrices.txt','r')

g = open('sensitivity_and_specificity.txt','w') 
"""
filename to write to without .txt
space
5 rows of confustion matrix
"""
with open('confusion_matrices.txt','r') as f:
    for line in f:
        line = line.rstrip()
        print("line: ", line)

