import csv

g = open("test.csv", 'w')
csv_out = csv.writer(g, delimiter=',')

x = [1,2,3,4]
y = [0,0,0,0]

for i in range(0, len(x)):
    csv_out.writerow([x[i],y[i]])

g.close()
