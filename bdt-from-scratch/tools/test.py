import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import distance
import numpy as np
import csv

#f = open("../data/modeBalanced/ModeBalanced_170_LIDC_809_Random.csv", 'r')
f = open("test.csv")
g = open("output.txt", 'w')

csv_f = csv.reader(f, delimiter=',')
#skip the header row
#csv_f.next()

"""
first run through of the file will save all of the
information on the nodules of low typicality

second run will calculate the distance from every
previously saved nodule to every other nodule
"""
x = []
y = []

for row in csv_f:
    x.append(float(row[0]))
    y.append(float(row[1]))

print("x: ", x)
print("y: ", y)

fig = plt.figure()

ax = fig.add_subplot(111)
ax.set_title('')
for i in range(0, len(x)):
    ax.plot([x[i]]*2, [-.5, .5], color="black")
ax.set_ylim([-10,10])


ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')

# remove the ticks from the top and right edges
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.show()
