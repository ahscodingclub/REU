import csv
import matplotlib.pyplot as plt

f = "typicality_file_1.csv"
csv_file = open(f)
csv_in = csv.reader(csv_file)

data = []

for row in csv_in:
    try:
      data.append(float(row[1]))
    except:
      print("error on: ", row[1])

print(len(data), data[0])

num_bins = 500

bins = [x/float(num_bins) for x in range(0,num_bins)]
(n,bins,patches) = plt.hist(data, bins)
plt.show()


"""
n - number of object in bin
bins - bin at that index
"""

points = []
for i in range(0, len(n)):
  if n[i]!=0:
    points.append([bins[i],n[i]])

csv_out = csv.writer(open("output.csv", 'w'))

for x in points:
  csv_out.writerow(x)


