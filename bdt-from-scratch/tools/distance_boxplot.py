import matplotlib.pyplot as plt
import csv

f = open("output.csv", 'r')
csv_in = csv.reader(f,delimiter=",")

distances = []
atyp = []

csv_in.next() #skip the header row
for row in csv_in:
    if float(row[1]) > (10**11):
        atyp.append(row[0])
    distances.append(float(row[1]))

g = open("atypical_nodules.csv", 'w')
csv_out = csv.writer(g)

csv_out.writerow(["ID"])
for x in atyp:
    csv_out.writerow([x])

plt.figure()
plt.boxplot(distances)
plt.show()
