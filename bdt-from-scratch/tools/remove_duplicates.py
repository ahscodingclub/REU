import csv

file_name = "atypical_nodules.csv"
column = 0

f = open(file_name, 'r')
csv_in = csv.reader(f)
csv_in.next()

l = []

for row in csv_in:
    l.append(float(row[column]))

f.close()

f = open(file_name, 'w')
csv_out = csv.writer(f)

#remove all duplicates
seen = set()
seen_add = seen.add
l = [x for x in l if not (x in seen or seen_add(x))]

print("length of l: ", len(l))
print("length of set l: ", len(set(l)))

for x in l:
    if l.count(x) > 1:
        print("x: ", x)


for row in l:
    csv_out.writerow([row])

f.close()
