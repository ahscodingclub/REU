import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
from sklearn.datasets import make_classification

#get the command line args
args = sys.argv[1:]
percent = 0
defin = 0
save_file = 0

if len(args) == 0:
    defin = int(input("Definition of typicality: \n1) Distance\n2) Typicality Rating\n3) Both\nchoice: "))
    percent = float(input("Percentile of atypicality: "))/100
elif len(args) <= 3:
    defin = int(args[0])
    percent = float(args[1])/100
    if len(args) == 3:
        save_file = args[2]
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

  data.append([float(row[9]),float(row[13])])

  # data.append([float(row[35]),float(row[25]),float(row[18]),float(row[20]),float(row[9]),float(row[26]),
  #              float(row[22]),float(row[24]),float(row[13])])


data = np.array(data)

mds = MDS(n_components=2, n_jobs=4)
pos = mds.fit(data.astype(np.float64)).embedding_
# Succeeds

similarities = euclidean_distances(data.astype(np.float32))

s = 100
typ = np.array([x[1] for x in zip(classes, pos) if x[0]==0])
atyp = np.array([x[1] for x in zip(classes, pos) if x[0]==1])

print(len(typ), len(typ[0]))
print(len(atyp), len(atyp[0]))

plt.scatter(typ[:, 0], typ[:, 1], color='turquoise', s=s, lw=0, label='typical')
plt.scatter(atyp[:, 0], atyp[:, 1], color='red', s=s, lw=0, label='atypical')
plt.legend(scatterpoints=1, loc='best', shadow=False)

if save_file != 0:
    plt.savefig(save_file)
else:
    plt.show()
