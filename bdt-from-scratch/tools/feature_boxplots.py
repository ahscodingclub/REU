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

atyp_nodule_ids = [642, 516, 1417, 139, 140, 525, 143, 144, 1041, 147, 405, 150, 281, 111, 157, 928, 417, 70, 166, 497, 170, 199, 116, 108, 178, 55, 53, 54, 311, 624, 58, 59, 60, 74, 319, 481, 196, 130, 327, 72, 73, 458, 75, 206, 81, 82, 83, 212, 213, 214, 57, 216, 89, 218, 1371, 222, 97, 98, 484, 101, 102, 103, 104, 161, 1132, 109, 61, 880, 753, 115, 372, 120, 84, 122, 94, 125, 85]

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
  plt.boxplot([typ_data[0], atyp_data[i]])
  plt.savefig(titles[i]+'.png')


