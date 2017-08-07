import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

"""
plot true negatives
plot unknown" up to 100-FP
plot true positives

list is in order of mean, median, mode
"""

false_positives = [100]*7
false_negatives = [100]*7

#100-FN
unknown_positives = [100-0,100-0,100-.06, 100-7.14,100-4.9,100-42.53,100-36.36]
#100-FP
unknown_negatives = [100-0,100-.07,100-.01, 100-15.63,100-5.08,100-2.86,100-22.5]

true_negatives = [ 90.63, 88.14, 90.48, 84.38, 40.68, 97.14, 74.16]
true_positives = [ 91.07, 88.24, 73.56, 92.86, 35.29, 57.47, 63.64]

plt.figure()
plt.subplots_adjust(hspace=.25, right=.8)

hfont = {'fontname':'Helvetica', 'size':28}
plt.title(' ', **hfont  )

#indices for mean median and mode
width = .1
o = width/2
negative_range = [o,width+o,2*width+o,3*width+o, 4*width+o, 5*width+o, 6*width+o]
positive_range = [8*width+o, 9*width+o, 10*width+o, 11*width+o, 12*width+o, 13*width+o, 14*width+o]

plt.bar(negative_range, false_negatives, width=width, color='r', hatch="xxx")
plt.bar(negative_range, unknown_negatives, width=width, color='gray')
plt.bar(negative_range, true_negatives, width=width, color='g', hatch="xxxxx")

plt.bar(positive_range, false_positives, width=width, color='r', hatch="xxx")
plt.bar(positive_range, unknown_positives, width=width, color='gray')
plt.bar(positive_range, true_positives, width=width, color='g', hatch="xxxxx")

#black bars
plt.bar([width, width*2, width*3, width*4, width*5, width*6, width*9, width*10, width*11, width*12, width*13, width*14], [100]*12, width=(width/20), color="black")

#x axis
hfont = {'fontname':'Helvetica', 'size':20}
#plt.xticks([width+o,width*7-o], ('Negatives', 'Positives'), **hfont)
plt.xticks([0],"")

green_patch = mpatches.Patch(color='green', label='Correct', hatch="///////////")
grey_patch = mpatches.Patch(color='grey', label='Indeterminate')
red_patch = mpatches.Patch(color='red', label='Incorrect')

plt.legend(handles=[green_patch, grey_patch, red_patch],
           bbox_to_anchor=(.99 , 1), prop={'size': 17})

plt.show()

"""
# Three subplots sharing both x/y axes
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(x, y)
axarr[0].set_title('Sharing X axis')
axarr[1].scatter(x, y)
"""
