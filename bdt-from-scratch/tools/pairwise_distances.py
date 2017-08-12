from scipy.spatial import distance
import matplotlib.pyplot as plt
from decimal import Decimal
import numpy as np
import csv

def remove_dups(l):
    prev = l[0][0]
    to_del = []
    for i in range(1,len(l)):
        if l[i][0] == prev:
            to_del.append(i-len(to_del))
        else:
            prev = l[i][0]

    for i in to_del:
        del l[i]
    return l

#######################
#Main Script
#######################
f = open("../data/modeBalanced/ModeBalanced_170_LIDC_809_Random.csv", 'r')
g = open("output.csv", 'w')

csv_in = csv.reader(f, delimiter=',')
csv_out = csv.writer(g, delimiter=',')
#skip the header row
row = next(csv_in)
# print("Circularity ",  row[13])
# print("Compactness: ",  row[9])
# print("Gabormean_0_2: ",  row[35])
# print("Intensity_difference: ",  row[25])
# print("Markov_1: ",  row[26])
# print("Max_Intensity: ",  row[18])
# print("Max_Intensity_BG: ",  row[22])
# print("SD_Intensity: ",  row[20])

nodule_ids = [6]
atyp_nodules = []
all_nodules = []
distances = []

#saving all nodules, and saving atypical
#nodules in a seperate list as well
for row in csv_in:
    atyp = False
    nod = []

    for x in row:
        try:
            var = float(x)
        except:
            var = 0
        nod.append(var)

    for nod_id in nodule_ids:
        if nod[0] == nod_id:
            atyp = True

    nod = [nod[0], nod[9], nod[18], nod[20], nod[22], nod[25], nod[26], nod[35]]
    all_nodules.append(nod)

    if atyp:
        atyp_nodules.append(nod)

#filter out unimportant features
"""
Compactness: 9
Circularity: 13Max_Intensity: 18
SD_Intensity: 20
Max_Intensity_BG: 22
Intensity_difference: 25
Markov_1: 26
Gabormean_0_2: 35
"""


for i in range(0, len(all_nodules)):
    distances.append(0)
    for nod in all_nodules:
        distances[i] = np.add(distances[i], distance.euclidean(nod,all_nodules[i]))

nodules = [[all_nodules[i][0], distances[i]] for i in range(0, len(all_nodules))]

#sorting nodules by id
nodules.sort(key=lambda x: x[0])
#removing duplicates
nodules = remove_dups(nodules)

csv_out.writerow(["ID", "Distances"])
for i in range(0, len(nodules)):
    csv_out.writerow([nodules[i][0], "{:.2E}".format(Decimal(nodules[i][1]))])

avg = np.sum(distances)/len(distances)

csv_out.writerow(["average: ", '%.2E' % Decimal(avg), "min: ", '%.2E' % Decimal(min(distances)), "max: ", '%.2E' % Decimal(max(distances))])
