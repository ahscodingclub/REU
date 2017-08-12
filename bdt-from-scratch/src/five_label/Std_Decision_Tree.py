"""
#####################################
CUSTOM BDT BUILD
Created on Wed Jul 06 13:54:20 2016

@authors: { Rachael Affenit <raffenit@gmail.com>,
            Erik Barns <erik.barns909@gmail.com> }
revamped by: { Jake Sauter <jsauter@oswego.edu> }

@INPROCEEDINGS {
  pele2008,
  title={A linear time histogram metric for improved sift matching},
  author={Pele, Ofir and Werman, Michael},
  booktitle={Computer Vision--ECCV 2008},

  pages={495--508},
  year={2008},
  month={October},
  publisher={Springer} }

#####################################
"""

#####################################
# DECISION TREE IMPLEMENTATION
#####################################
# General Imports
from __future__ import print_function
import math # Used for logarithm and sqrt
import scipy # Used for integration
import pydot # Plotting decision trees
import pandas as pd # Don't delete this. Just don't.
import numpy as np # Numpy arrays
import matplotlib
from sklearn.tree import DecisionTreeClassifier
import numpy as np

#set the display to default display
#so no errors when run on a server
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import csv # Read and write csv files
import time # for testing purposes
import copy
from sklearn.metrics import confusion_matrix # Assess misclassification
from sklearn.metrics import accuracy_score
import ast
import sys
import warnings

# silence warnings for old sklearn kfold
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn.cross_validation import KFold
#####################################
# GATHERING DATA
#####################################
# IMPORT DATA
def importIdData(set_name):
    global set_data_array
    global set_header

    id_data  = pd.read_csv(set_name, header =0)
    set_header = list(id_data.columns.values)
    id_data = id_data._get_numeric_data()
    set_data_array = id_data.as_matrix()

def importAllData(set_name):
    global LIDC_Data
    global header

    LIDC_Data  = pd.read_csv(set_name, header =0)
    header = list(LIDC_Data.columns.values)[:-4]
    LIDC_Data = LIDC_Data._get_numeric_data()
    LIDC_Data = LIDC_Data.as_matrix()

## SET TRAINING DATA
def setTrain(trn_data_array):
    global train_features
    global calib_features

    split = int((6.0/7.0) * len(trn_data_array))
    train_features = trn_data_array[:split].tolist() # training data
    calib_features = trn_data_array[split:].tolist() # calibration data

# EXTRACT DATA
def getPigns(dataset):
    nodePigns = []
    # For each case at this node, calculate a vector of probabilities
    for case in dataset:
        currentLabel = case[-4:]
        zeroratings = 0
        pign = [0]*5

        # Convert radiologist ratings into pignistic probability distributions

        if pign_type == 1:      #mean
            sum = 0
            for i in currentLabel:     # Count number of instances of each rating
                sum += i
            mean = sum/4
            pign[int(math.floor(mean))-1] = 1 - (mean - math.floor(mean))
            if mean-int(math.floor(mean)) != 0:
                pign[int(math.floor(mean))] = mean - math.floor(mean)

        elif pign_type == 2:    #median
            median = getMedian(currentLabel)
            if float(median).is_integer():
                pign[int(math.floor(median))-1] = 1
            else:
                pign[int(math.floor(median))-1] = 1 - (median - math.floor(median))
                if median-math.floor(median) != 0:
                    pign[int(math.floor(median))] = median - math.floor(median)

        elif pign_type == 3:    #mode (which appears most often, majority vote)
            mode = getMode(currentLabel)
            pign[mode-1] = 1

        elif pign_type == 4:    #distribution
            pign = []
            for i in range (0, 6): # Count number of instances of each rating
                if i == 0:
                    zeroratings += currentLabel.count(i)
                else:
                    pign.append(currentLabel.count(i))
            pign[:] = [float(x) / (4 - zeroratings) for x in pign]

        nodePigns.append(pign) # Add pign to list of pigns
    return nodePigns

#####################################
# CREATING THE BDT
#####################################
def createTree(train, train_labels):
    clf=DecisionTreeClassifier(criterion = "entropy", splitter="best", min_samples_leaf=12)
    clf.classes = [1,2,3,4,5]
    print("classes: ", clf.classes)
    clf.fit(train,train_labels)
    return clf

#####################################
# EVALUATION METHODS
#####################################
def getMean(lst):
    sum = 0
    for i in lst:     # Count number of instances of each rating
        sum += i
        mean = sum/4
    return mean

def getMedian(case):
    case.sort()
    while(len(case) > 2):
        case = case[1:-1]
        if len(case) == 1:
            return case[0]
        else:
            return round((float(case[0])+case[1])/2)

def getMode(case):
    counts = [0]*5
    for vote in case:
        counts[int(vote)-1]+=1
    return int(round(getMax(counts))+1)

def getMax(lst):
    mx = max(lst)
    mx_vals = []

    for k,x in enumerate(lst):
        if x == mx:
            mx_vals.append(k)
    if len(mx_vals) == 1:
        return mx_vals[0]
    else:
        return (sum(mx_vals)/len(mx_vals))

#####################################
# MAIN SCRIPT: Build, Classify, Output
#####################################
# Setup
#input loop for PLV settings
#arguments to script are [pignisitic type(1-4), output comparison type(1-4), output file name(string), testing/traing(y/n)]
args = sys.argv[1:]
print("args: ", args)

global output_type

if len(args) == 0:
    pign_type = None
    while pign_type != 1 and pign_type != 2 and pign_type != 3 and pign_type != 4:
        pign_type = int(input("\n\nPignistic Type?\n1.Mean\n2.Median\n3.Mode\n\ntype: "))

    # file output settings
    f_name = raw_input("\n\nfile for confusion matrix: ")

    #input loop for variable settings
    var_set = None
    while var_set != "y" and var_set != "n":
        var_set = raw_input("\n\ntesting?(y/n): ")

elif len(args) == 3:
    pign_type = int(args[0])
    f_name = args[1]
    var_set = args[2]
else:
    print("arguments to script are [pignisitic type(1-4), output file name(string), testing/traing(y/n)]")
    sys.exit()

#open the output file
f = open(f_name, "w")

importIdData("../../data/clean/LIDC_809_Complete.csv")

kfolds = 6
nparent = 24
nchild = 12
maxdepth = 25

if(var_set == "n"):
    importAllData("../../data/modeBalanced/ModeBalanced_170_LIDC_809_Random.csv")
elif(var_set == "y"):
    importAllData("../../data/modeBalanced/testing_file.csv")

###### K-FOLD VALIDATION ######
kf = KFold(len(LIDC_Data), kfolds)

k_round = 1
k_best = [None]*7

global graph
global typicality_list
global id_list

typicality_list = []
id_list = []

"""
manually setting kfolds so every case is tested
if var_set == "n":
  kf = [[[range(142,850)],[range(0,142)]],[[range(0,142),range(284,850)],[range(142,284)]],[[range(0,284),range(426,850)],[range(284,426)]],[[range(0,426), range(568,850)],[range(426,568)]],[[range(0,568), range(710,850)],[range(568,710)]],[[range(0,710)],[range(710,850)]] ]
"""


for trn_ind, tst_ind in kf:
    trainLabels = []
    testLabels = []
    setTrain(LIDC_Data[trn_ind])
    test_features = LIDC_Data[tst_ind].tolist()
    for i in range(0, len(test_features)):
      id_list.append(test_features[i][0])
      test_features[i] = test_features[i][1:]

    for i in range(0, len(train_features)):
        train_features[i] = train_features[i][1:]

    # Console Output
    print("\n K-FOLD VALIDATION ROUND ",k_round," OF ",kfolds)
    print("#################################")
    print("Train Size: ", len(train_features))
    print("Test Size: ", len(test_features))
    print ("Building Decision Tree...")

    # Create Tree
    # setting "switch = True" will make new tree each time
    train_labels = []
    test_labels = []

    for i in range(0, len(train_features)):
        votes = [int(x) for x in train_features[i][-4:]]
        if pign_type == 1:
            label = getMean(votes)
        elif pign_type == 2:
            label = getMedian(votes)
        elif pign_type == 3:
            label = getMode(votes)
        train_labels.append(label)
        train_features[i] = train_features[i][:-4]

    tree = createTree(train_features, train_labels)



    for i in range(0, len(test_features)):
        votes = [int(x) for x in test_features[i][-4:]]
        if pign_type == 1:
            label = getMean(votes)
        elif pign_type == 2:
            label = getMedian(votes)
        elif pign_type == 3:
            label = getMode(votes)
        test_labels.append(label)
        test_features[i] = test_features[i][:-4]

    print(train_labels)
    print(test_labels)

    print ("Classifying Training Set...")
    train_predicted = tree.predict(copy.deepcopy(train_features))

    print(train_predicted)

    print(confusion_matrix(train_labels, train_predicted, [1,2,3,4,5]))
    print("accuracy: ", accuracy_score(train_labels,train_predicted))

   # Classify testing set
    print ("Classifying Testing Set...")
    test_predicted = tree.predict(test_features)

    conf_mat = confusion_matrix(test_labels, test_predicted, [1,2,3,4,5])
    accuracy = accuracy_score(test_labels,test_predicted)

    if accuracy > k_best[1]:
        k_best = [k_round, accuracy, conf_mat]

    # increase round of k-fold vlaidation
    k_round += 1

    importances = tree.feature_importances_
    std = np.std(tree.feature_importances_,axis=0)
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")

    for f in range(10):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(10), importances[indices],
           color="r",  align="center")
    plt.xticks(range(10), indices)
    plt.xlim([-1, 10])
    plt.show()

# Output data to text files
accuracy = k_best[1]
conf_mat = k_best[2]

print ("\nWriting Data for best fold k =", k_best[0], "...\n")

# write training data
print("confusion matrix: \n", conf_mat, file=f)
print("accuracy: \n", accuracy, file=f)
print("unique cases: ", len(set(id_list)), file=f)

# Close output file
f.close()
