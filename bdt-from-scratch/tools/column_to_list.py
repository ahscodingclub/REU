import csv

file_name = "atypical_nodules.csv"
column = 0

f = open(file_name, 'r')
csv_in = csv.reader(f)
csv_in.next() #skip the header row

l = []

for row in csv_in:
    print("x: ", row[column])
    l.append(float(row[column]))

g = open("test.txt", 'w')

l = set(l)
print(len(l), "\n\n")

print(l)

f.close()
g.close()
