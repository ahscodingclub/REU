import csv
import matplotlib.pyplot as plt

f = "pairwise_distances.csv"
csv_file = open(f)
csv_in = csv.reader(csv_file)

data = []

csv_in.next() #skip headers
for row in csv_in:
    data.append(float(row[1]))

print(len(data), data[0])

plt.hist(data)
plt.show()
f.close()
