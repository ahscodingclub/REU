import csv
import matplotlib.pyplot as plt
import numpy as np
import re
import sys

titles = ['Area', 'ConvexArea', 'Perimeter', 'ConvexPerimeter', 'EquivDiameter', 'MajorAxisLength', 'MinorAxisLength', 'Elongation', 'Compactness', 'Eccentricity', 'Solidity', 'Extent', 'Circularity', 'RadialDistanceSD', 'SecondMoment', 'Roughness', 'MinIntensity', 'MaxIntensity', 'MeanIntensity', 'SDIntensity', 'MinIntensityBG', 'MaxIntensityBG', 'MeanIntensityBG', 'SDIntensityBG', 'IntensityDifference', 'markov1', 'markov2', 'markov3', 'markov4', 'markov5', 'gabormean_0_0', 'gaborSD_0_0', 'gabormean_0_1', 'gaborSD_0_1', 'gabormean_0_2', 'gaborSD_0_2', 'gabormean_1_0', 'gaborSD_1_0', 'gabormean_1_1', 'gaborSD_1_1', 'gabormean_1_2', 'gaborSD_1_2', 'gabormean_2_0', 'gaborSD_2_0', 'gabormean_2_1', 'gaborSD_2_1', 'gabormean_2_2', 'gaborSD_2_2', 'gabormean_3_0', 'gaborSD_3_0', 'gabormean_3_1', 'gaborSD_3_1', 'gabormean_3_2', 'gaborSD_3_2', 'Contrast', 'Correlation', 'Energy', 'Homogeneity', 'Entropy', 'x_3rdordermoment', 'Inversevariance', 'Sumaverage', 'Variance', 'Clustertendency', 'MaxProbability']

atyp_data = [[]*len(titles)]
typ_data = [[]*len(titles)]

"""
Recieving the command line arguments and
and removing the program name
"""
args = sys.argv[1:]

if len(args) == 0:
    defin = int(input("Definition of typicality: \n1) Distance\n2) Typicality Rating\n3) Both\nchoice: "))
    percent = float(input("Percentile of atypicality: "))/100
elif len(args) == 2:
    defin = int(args[0])
    percent = float(args[1])/100
else:
    print("Command line arguments should contain: [Definition of typicality(1/2), Percentile of typicality(0-100)]")

"""
Accessing the nodules IDs, presorted by
typicality rating
"""
nodule_ids = []
file_name = "atypical_nodules.csv"
f = open(file_name)
csv_in = csv.reader(f)
csv_in.next()
for row in csv_in:
    nodule_ids.append([float(row[0]), float(row[2])])
number_of_nodules = int(percent*len(nodule_ids))
#setting appropriate atypical nodule ids
atyp_nodule_ids = []

"""
Isolating the Xth percentile of nodules from the rest
and defining them as atypical nodules
"""
if defin == 1:
    atyp_nodule_ids = [x[0] for x in nodule_ids[0:number_of_nodules]]
    defin = "distance"
elif defin == 2:
    atyp_nodule_ids = [x[1] for x in nodule_ids[0:number_of_nodules]]
    defin = "typicality"
elif defin == 3:
    atyp_nodule_ids = set([x[0] for x in nodule_ids[0:number_of_nodules]]).intersection(set(x[1] for x in nodule_ids[0:number_of_nodules]))
    defin = "both"

print("atyp: ", atyp_nodule_ids)
f = open("../data/modeBalanced/ModeBalanced_170_LIDC_809_Random.csv", 'r')
csv_f = csv.reader(f, delimiter=",")
csv_f.next()

"""
Acessing the associated features of precaclulated
typical and atypical nodule ids
"""
for row in csv_f:
  atyp = False
  for x in atyp_nodule_ids:
    if float(row[0]) == x:
      atyp = True

      for i in range(0, len(titles)):
        if atyp:
            atyp_data[i].append(float(row[i+1]))

        else:
            typ_data[i].append(float(row[i+1]))

for i in range(0, len(atyp_data)):
  plt.figure()
  plt.title(titles[i])
  plt.hist([typ_data[i], atyp_data[i]])
  plt.savefig(str(defin)+"_"+str(percent)+"_"+titles[i]+"_hist"+'.png')


for i in range(0, len(atyp_data)):
  plt.figure()
  plt.title(titles[i])
  plt.boxplot([typ_data[i], atyp_data[i]])
  plt.savefig(str(defin)+"_"+str(percent)+"_"+titles[i]+'.png')
