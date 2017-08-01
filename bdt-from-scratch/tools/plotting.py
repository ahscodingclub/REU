import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

"""
plot true negatives
plot unknown" up to 100-FP
plot true positives

list is in order of mean, median, mode
"""

false_negatives = [100]*4
false_positives = [100]*4

unknown_negatives = [100-5.26,100-6.06,100-12.73,100-23.35]
unknown_positives = [100-4.17,100-12.5,100-6.25,100-23.05] 

true_negatives = [39.47,48.48,56.36,47.84]
true_positives = [64.58,66.07,62.5,53.85]

"""
unknown_negatives = [100-1.42,100-4.97,100-2.13] 
unknown_positives = [100-1.42,100-1.42,100-4.97]

true_negatives = [39,48,56]
true_positives = [65,66,63]
"""

plt.figure()       
plt.subplots_adjust(hspace=.25, right=.8)

hfont = {'fontname':'Helvetica', 'size':28}
plt.title(' ', **hfont  ) 

#indices for mean median and mode 
width = .1
o = width/2
negative_range = [o,width+o,2*width+o,3*width+o]
positive_range = [5*width+o, 6*width+o, 7*width+o, 8*width+o]

plt.bar(negative_range, false_negatives, width=width, color='r', hatch="\\\\\\\\")
plt.bar(negative_range, unknown_negatives, width=width, color='gray')
plt.bar(negative_range, true_negatives, width=width, color='g', hatch="////")

plt.bar(positive_range, false_positives, width=width, color='r', hatch="\\\\\\\\")
plt.bar(positive_range, unknown_positives, width=width, color='gray')
plt.bar(positive_range, true_positives, width=width, color='g', hatch="////")

#black bars
plt.bar([width, width*2, width*3, width*6, width*7, width*8], [100]*6, width=(width/20), color="black")

#x axis
hfont = {'fontname':'Helvetica', 'size':20}
plt.xticks([width+o,width*7-o], ('Negatives', 'Positives'), **hfont)

green_patch = mpatches.Patch(color='green', label='True', hatch="///////////")
grey_patch = mpatches.Patch(color='grey', label='Indistinguished')
red_patch = mpatches.Patch(color='red', label='False')

plt.legend(handles=[green_patch, grey_patch, red_patch], 
           bbox_to_anchor=(1.0001 , 1), prop={'size': 17})

plt.show()

"""
# Three subplots sharing both x/y axes
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(x, y)
axarr[0].set_title('Sharing X axis')
axarr[1].scatter(x, y)
"""
