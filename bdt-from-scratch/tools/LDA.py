import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances

#get the command line args
args = sys.argv[1:]

if len(args) == 0:
    defin = int(input("Definition of typicality: \n1) Distance\n2) Typicality Rating\n3) Both\nchoice: "))
    percent = float(input("Percentile of atypicality: "))/100
elif len(args) == 2:
    defin = int(args[0])
    percent = float(args[1])/100
else:
    print("Command line arguments should contain: [Definition of typicality(1/2), Percentile of typicality(0-100)]")

nodule_ids = []
file_name = "atypical_nodules.csv"
f = open(file_name)
csv_in = csv.reader(f)
csv_in.next()
for row in csv_in:
    nodule_ids.append([float(row[0]), float(row[2])])
f.close()
number_of_nodules = int(percent*len(nodule_ids))
#setting appropriate atypical nodule ids

atyp_nodule_ids = []

if defin == 1:
    atyp_nodule_ids = [x[0] for x in nodule_ids[0:number_of_nodules]]
    defin = "distance"
elif defin == 2:
    atyp_nodule_ids = [x[1] for x in nodule_ids[0:number_of_nodules]]
    defin = "typicality"
elif defin == 3:
    atyp_nodule_ids = set([x[0] for x in nodule_ids[0:number_of_nodules]]).intersection(set(x[1] for x in nodule_ids[0:number_of_nodules]))
    defin = "both"

f = open("../data/modeBalanced/ModeBalanced_170_LIDC_809_Random.csv", 'r')
csv_in = csv.reader(f, delimiter=",")
csv_in.next()

classes = []
data = []
for row in csv_in:
  atyp = False
  for x in atyp_nodule_ids:
    if float(row[0]) == x:
      atyp = True

  if atyp:
      classes.append(1)
  else:
      classes.append(0)

  data.append([float(row[35]),float(row[25]),float(row[18]),float(row[20]),float(row[9]),float(row[26]),
               float(row[22]),float(row[24]),float(row[13])])

data = np.array(data)

lda = LinearDiscriminantAnalysis(n_components=4)

pos = lda.fit_transform(data, classes)

print("COEFS")
for x in lda.coef_:
    print(x)

print("\nSCALINGS")
for x in lda.scalings_:
    print(x)

g = open("output.csv", "w")
csv_out = csv.writer(g)

for x in pos:
    csv_out.writerow(x)

plt.figure()
atyp = pos[0:number_of_nodules]
typ = pos[number_of_nodules:]
plt.scatter(atyp,[1]*len(atyp), color='red', label='atypical')
plt.scatter(typ,[0]*len(typ), color='turquoise', label="typical")

plt.show()
