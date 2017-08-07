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
from sklearn.model_selection import train_test_split
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
    global train_cases
    global calib_features

    split = int((6.0/7.0) * len(trn_data_array))
    train_cases = trn_data_array[:split].tolist() # training data
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
def createTree(train, train_predicted):
    clf=DecisionTreeClassifier(criterion = "entropy", splitter="best", min_samples_leaf=12)
    clf.classes = [1,2,3,4,5]
    print("classes: ", clf.classes)
    clf.fit(train,train_predicted)
    return clf

#####################################
# EVALUATION METHODS
#####################################
def getActualLabel(actual):
    if pign_type == 1:
        labels = getMean(actual)
    elif pign_type == 2:
        labels = getMedian(actual)
    elif pign_type == 3:
        labels = getMode(actual)
    return labels


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

def getMean(lst):
    sum = 0
    for i in lst:     # Count number of instances of each rating
        sum += i
        mean = sum/4
    return mean

def getPredictedLabels(predicted):
    #mean
    if pign_type == 1:
        labels = [int(round(1*case[0]+2*case[1]+3*case[2]+4*case[3]+5*case[4])) for case in predicted]

    #median
    elif pign_type == 2:
        labels = []
        for case in predicted:
            pred_index = 0
            prob_total = 0
            while prob_total <= .5:
                prob_total += case[pred_index]
                pred_index += 1
            labels.append(int(round(pred_index)))

    #mode
    elif pign_type == 3:
        labels = [int(round(getMax(case)+1)) for case in predicted]

    return labels

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

nparent = 24
nchild = 12
maxdepth = 25

if(var_set == "n"):
    importAllData("../../data/modeBalanced/ModeBalanced_170_LIDC_809_Random.csv")
elif(var_set == "y"):
    importAllData("../../data/modeBalanced/testing_file.csv")

all_data = LIDC_Data.tolist()

#splitting data
train_cases, test_cases = train_test_split(all_data, test_size=0.3, random_state=42)

global id_list

id_list = []

all_data = LIDC_Data.tolist()

#splitting data
train_cases, test_cases = train_test_split(all_data, test_size=0.3, random_state=42)

for i in range(0, len(train_cases)):
    id_list.append(train_cases[i][0])
    train_cases[i] = train_cases[i][1:]

for i in range(0, len(test_cases)):
    id_list.append(test_cases[0])
    test_cases[i] = test_cases[i][1:]

train_actual = []
train_predicted = []
test_actual = []
test_predicted = []

train_actual = [int(getActualLabel(x[-4:])) for x in train_cases]
test_actual = [int(getActualLabel(x[-4:])) for x in test_cases]

# Console Output
print("#################################")
print("Train Size: ", len(train_cases))
print("Test Size: ", len(test_cases))
print ("Building Belief Decision Tree...")

train_features = [x[:-4] for x in train_cases]
test_features = [x[:-4] for x in test_cases]

tree = createTree(train_cases, train_actual)

print ("Classifying Training Set...")
train_predicted = tree.predict(train_cases)

print(confusion_matrix(train_predicted, train_predicted, [1,2,3,4,5]))
print("accuracy: ", accuracy_score(train_predicted,train_predicted))

# Classify testing set
print ("Classifying Testing Set...")
test_predicted = tree.predict(test_cases)

conf_mat = confusion_matrix(test_actual, test_predicted, [1,2,3,4,5])
accuracy = accuracy_score(test_actual,test_predicted)

# write training data
print("confusion matrix: \n", conf_mat, file=f)
print("accuracy: \n", accuracy, file=f)
print("\n\ntest_act: ", test_actual, file=f)
#print("unique cases: ", len(set(id_list)), file=f)

# Close output file
f.close()
