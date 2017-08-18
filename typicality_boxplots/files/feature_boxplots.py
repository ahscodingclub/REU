import csv
import matplotlib.pyplot as plt
import numpy as np
import re

"""
FEATURES
gabormean_0_2
IntensityDifference
MaxIntensity
SDIntensity
Compactness
markov1
MaxIntensityBG
SDIntensityBG
Circularity
"""

atyp_data = [[],[],[],[],[],[],[],[],[]]
typ_data = [[],[],[],[],[],[],[],[],[]]

defin = int(input("Definition of typicality: \n1) Distance\n2) Typicality Rating\n3) Both\nchoice: "))


f = open("../data/modeBalanced/ModeBalanced_170_LIDC_809_Random.csv", 'r')
csv_f = csv.reader(f, delimiter=",")
csv_f.next()

for row in csv_f:
  atyp = False
  for x in atyp_nodule_ids:
    if float(row[0]) == x:
      atyp = True

  if atyp:
    atyp_data[0].append(float(row[35]))
    atyp_data[1].append(float(row[25]))
    atyp_data[2].append(float(row[18]))
    atyp_data[3].append(float(row[20]))
    atyp_data[4].append(float(row[9]))
    atyp_data[5].append(float(row[26]))
    atyp_data[6].append(float(row[22]))
    atyp_data[7].append(float(row[24]))
    atyp_data[8].append(float(row[13]))

  else:
    typ_data[0].append(float(row[35]))
    typ_data[1].append(float(row[25]))
    typ_data[2].append(float(row[18]))
    typ_data[3].append(float(row[20]))
    typ_data[4].append(float(row[9]))
    typ_data[5].append(float(row[26]))
    typ_data[6].append(float(row[22]))
    typ_data[7].append(float(row[24]))
    typ_data[8].append(float(row[13]))

titles = ["gabormean_0_2", "IntensityDifference", "MaxIntensity", "SDIntensity", "Compactness", "markov1", "MaxIntensityBG", "SDIntensityBG", "Circularity"]

for i in range(0, len(atyp_data)):
  plt.figure()
  plt.title(titles[i])
  plt.boxplot([typ_data[i], atyp_data[i]])
  plt.savefig(titles[i]+'.png')
