import csv
import matplotlib.pyplot as plt
import numpy as np

nodules = []

f = open("../data/modeBalanced/ModeBalanced_170_LIDC_809_Random.csv", 'r')
csv_f = csv.reader(f, delimiter=",")
titles = next(csv_f)[1:-4]

malig = []
benign = []
for row in csv_f:
    row = [float(x) for x in row]
    mean = (row[-4]+row[-3]+row[-2]+row[-1])/4
    row = row[1:-4]
    if mean > 3.5:
        malig.append(row)
    else:
        benign.append(row)

for i in range(0, len(titles)):
  plt.figure()
  plt.title(titles[i])
  plt.hist([[x[i] for x in benign],[x[i] for x in malig]], label=["Benign","Malignant"])
  plt.legend()
  plt.savefig(titles[i]+"_distribution.png")
