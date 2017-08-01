from scipy.spatial import distance
import numpy as np
import csv 

f = open("../data/modeBalanced/ModeBalanced_170_LIDC_809_Random.csv", 'r')
g = open("output.txt", 'w')

csv_f = csv.reader(f, delimiter='\t')
#skip the header row
csv_f.next() 

"""
first run through of the file will save all of the 
information on the nodules of low typicality

second run will calculate the distance from every 
previously saved nodule to every other nodule
"""
nodule_ids = [6]
atyp_nodules = []
all_nodules = []
distances = [0]*len(nodule_ids)

for row in csv_f: 
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

    all_nodules.append(nod)

    if atyp:
        atyp_nodules.append(nod)

for nod in all_nodules:
    for i in range(0, len(nodule_ids)):
        distances[i] = np.add(distances[i], distance.euclidean(nod,atyp_nodules[i]))


for i in range(0, len(nodule_ids)):
    print("id: ", nodule_ids[i], " , sum of distances: ", distances[i])

#now need to calculate distances from every other nodule to every nodule, 
#for now I will probably take an average to see where the atypical cases fall
#in regards to the average
distances = []

for nod_1 in all_nodules:
  for nod_2 in all_nodules:
    distances.append(distance.euclidean(nod_1,nod_2))

avg = np.sum(distances)/len(distances)

print("average: ", avg)
