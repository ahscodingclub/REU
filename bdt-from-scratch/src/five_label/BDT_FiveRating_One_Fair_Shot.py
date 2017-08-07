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
# BELIEF DECISION TREE IMPLEMENTATION
#####################################
# General Imports
from __future__ import print_function
import math # Used for logarithm and sqrt
import scipy # Used for integration
import pydot # Plotting decision trees
import pandas as pd # Don't delete this. Just don't.
import numpy as np # Numpy arrays
import matplotlib

#set the display to default display
#so no errors when run on a server
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import csv # Read and write csv files
import time # for testing purposes
import copy
from sklearn.metrics import confusion_matrix # Assess misclassification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy import spatial # Cosine similarity
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

    LIDC_Data  = pd.read_csv(set_name, header=0)
    header = list(LIDC_Data.columns.values)[:-4]
    LIDC_Data = LIDC_Data._get_numeric_data()
    LIDC_Data = LIDC_Data.as_matrix()

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
# CALCULATE ENTROPY OF A NODE
def calcEntrop(dataset):
    # For each case at this node, calculate a vector of probabilities
    nodePigns = getPigns(dataset)

    # Compute average pignistic probability
    Apign = [0.0, 0.0, 0.0, 0.0, 0.0]
    for k in range (0, len(nodePigns)): # Sum of probabilities for each rating
        for i in range (0, 5):
            Apign[i] += nodePigns[k][i]
    Apign[:] = [x / len(nodePigns) for x in Apign]

    # Compute entropy
    entrop = 0
    multList = []
    logList = []
    for y in Apign:
        if (y == 0):
            logList.append(0)
        else:
            logList.append(math.log(y, 2))

    # For each class in log(Apign) and Apign, multiply them together and find the entropy
    for k in range (0, 5):
        multList.append(Apign[k] * logList[k])
    entrop = - sum(multList)

    return entrop, Apign, nodePigns

# SPLITTING FUNCTION
# Split dataset into two sets with values for splitFeature higher and lower than the splitValue

#TODO could be a fundamental error here?
def splitData(dataset, splitFeat, splitValue):
    lowSet = []
    highSet = []

    # Split dataset into two sets with values for splitFeature higher and lower than the splitValue
    for case in dataset:
        if case[splitFeat] < splitValue:
            lowSet.append(case)
        elif case[splitFeat] >= splitValue:
            highSet.append(case)
    return lowSet, highSet

# DETERMINE BEST FEATURE
#TODO could be a fundamental error here?
def bestFeatureSplit(dataset, nFeat, min_parent, min_child):
    numFeat = nFeat # 63-64 features
    baseEntrop, baseApign, basePigns = calcEntrop(dataset) # Get information from parent node
    bestGain = 0.0
    bestFeature = 0
    bestValue = 0
    global x

    # For all features
    for j in range(0,numFeat):
        uniqueVals = []
        uniqueVals[:] = [case[j] for case in dataset] # All possible values of feature

        # For all possible values
        for val in uniqueVals:
            subDataset = splitData(dataset, j, val) # Splits data
            newEntrop = 0.0
            splitInfo = 0.0
            gainRatio = 0.0
            lowSet = subDataset[0]
            highSet = subDataset[1]

            # If relevant leaf conditions are met
            if (len(lowSet) >= min_child) and (len(highSet) >= min_child) and (len(dataset) >= min_parent):

                # Calculate newEntropy and splitInfo sums over both datasets
                for i in range (0, 2):
                    prob = abs(float(len(subDataset[i]))) / abs(float(len(dataset[i])))
                    prob2 = abs(float(len(subDataset[i]))) / float(len(dataset[i]))

                    entrop = calcEntrop(subDataset[i])[0]

                    newEntrop += prob * entrop # Sum over both datasets
                    splitInfo += prob * math.log(prob2, 2)

                # Calculate Gain Ratio
                infoGain = baseEntrop - newEntrop
                if splitInfo != 0:
                    gainRatio = float(infoGain) / (-1*splitInfo)
                else:
                    gainRatio = 0

                if(gainRatio > bestGain):
                    bestGain = gainRatio
                    bestFeature = j
                    bestValue = val

    return bestFeature, bestValue, bestGain, baseApign, basePigns

# DETERMINE IF ALL LIST ITEMS ARE THE SAME
def all_same(items):
    return all(x == items[0] for x in items)

# CREATING THE TREE RECURSIVELY
#TODO could be a fundamental error here?
def createTree(dataset, labels, min_parent, min_child, curr_depth, max_depth):
    # If labels exist in this subset, determine best split
    if len(labels) >= 1:

        # Get Best Split Information
        output = bestFeatureSplit(dataset, len(labels), min_parent, min_child) # index of best feature
        bestFeat = output[0]
        bestVal = output[1]
        bestGainRatio = output[2]
        baseApign = output[3]
        basePigns = output[4]

        # Get label of best feature
        bestFeatLabel = labels[bestFeat]

        # Create root node
        decision_tree = {bestFeatLabel:{"BBA":baseApign}}
        del(labels[bestFeat]) #remove chosen label from list of labels

        # Stopping Conditions
        if (bestGainRatio == 0) and (bestFeat == 0) and (bestVal == 0):
            decision_tree = {"Leaf":{"BBA":baseApign}}
            return decision_tree
        elif (bestGainRatio == 0):
            decision_tree = {"Leaf":{"BBA":baseApign}}
            return decision_tree
        elif (all_same(basePigns)):
            decision_tree = {"Leaf":{"BBA":baseApign}}
            return decision_tree
        elif (curr_depth == max_depth):
            decision_tree = {"Leaf":{"BBA":baseApign}}
            return decision_tree

        # Recursive Call
        else:
            # Create a split
            decision_tree[bestFeatLabel][bestVal] = {}
            subLabels = labels[:]
            lowSet, highSet = splitData(dataset, bestFeat, bestVal) # returns low & high set

            # Recursively create child nodes
            decision_tree[bestFeatLabel][bestVal]["left"] = createTree(lowSet, subLabels, min_parent, min_child,\
                                                                       curr_depth+1, max_depth)
            decision_tree[bestFeatLabel][bestVal]["right"] = createTree(highSet, subLabels, min_parent, min_child,\
                                                                        curr_depth+1, max_depth)
            return decision_tree

    # If no labels left, then STOP
    elif (len(labels) < 1):
        return

#####################################
# CLASSIFY NEW CASES
#####################################
#TODO could be a fundamental error here?
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]

    if (firstStr == 'Leaf'): # IF LEAF NODE, RETURN BBA VECTOR
        bba = inputTree['Leaf'].values()[0]
        return bba

    elif (firstStr == 'right' or firstStr == 'left'): # IF RIGHT OR LEFT, RECURSIVE CALL ON SUBTREE OF THAT NODE
        return classify(inputTree[firstStr],featLabels,testVec)

    else: # key is integer or feature label
        secondDict = inputTree[firstStr]
        keys = secondDict.keys()
        featIndex = featLabels.index(firstStr)
        if (keys[0] == 'BBA'): # IF key is BBA, CLASSIFY NEXT KEY IN TREE
            key = keys[1]
        else:
            key = keys[0] # KEY IS SPLIT VALUE

        if testVec[featIndex] > key: #IF GREATER THAN GET SUBTREE RIGHT ELSE GET SUBTREE LEFT
            k = secondDict[key].keys()[0]
            return classify(secondDict[key][k],featLabels,testVec) # GO RIGHT
        else: # GET SUBTREE
            k = secondDict[key].keys()[1]
            return classify(secondDict[key][k],featLabels,testVec) # GO LEFT

#####################################
# EVALUATION METHODS
#####################################

# calculate a confusion matrix given predicted and actual, using full complexity of BBA
# returns one confusion matrix
#def getConfusionMatrix(predicted,actual):
#  return cross_product(predicted,actual)

def getActualLabel(actual):
    if pign_type == 1:
        labels = getMean(actual)
    elif pign_type == 2:
        labels = getMedian(actual)
    elif pign_type == 3:
        labels = getMode(actual)
    elif pign_type == 4:
        labels = getDist(actual)
    return labels

def getDist(case):
    pign = []
    zeroratings = 0
    for i in range (0, 6): # Count number of instances of each rating
        if i == 0:
            zeroratings += case.count(i)
        else:
            pign.append(case.count(i))
    pign[:] = [float(x) / (4 - zeroratings) for x in pign]
    return pign



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

#returns the index of the maximum element
#in a list
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
    labels = predicted
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
    elif pign_type == 3 or pign_type == 4:
        labels = [int(round(getMax(case)+1)) for case in predicted]

    return labels

def getAccuracy(class_matrix):
    accy = 0.0
    for j in range(0,5):
        accy += class_matrix[j][j]
    return 100 * float(accy) / sum2D(class_matrix)

def sum2D(input):
    return sum(map(sum, input))

def getConfusionMatrix(predicted, actual):
    conf_mat = [[0,0,0,0,0],
                [0,0,0,0,0],
                [0,0,0,0,0],
                [0,0,0,0,0],
                [0,0,0,0,0]]
    for i in range(0,len(predicted)):
        conf_mat = np.add(conf_mat, cross_product(predicted[i],actual[i]))
    return conf_mat

def cross_product(X,Y):
    product = [ [ 0 for y in range(len(Y)) ] for x in range(len(X)) ]
    # iterate through rows of X
    for i in range(len(X)):
        # iterate through rows of Y
        for j in range(len(Y)):
            product[i][j] += X[i] * Y[j]
    return product

#####################################
# DRAW THE DECISION TREE
#####################################

# IF tree in file is same tree we want, USE IT rather than building it
def getTrees(tf,head,np,nc,mind,md, switch):
    param = [np,nc,md]
    if(switch):
         with open("../../output/tree.txt","wb") as t:
            trees = [param,createTree(train_cases, header, nparent, nchild, mind, maxdepth)]
            t.write(trees.__repr__())
            return trees[1]
    else:
        with open("../output/tree.txt","r") as t:
            tree_file = t.read()

        clear = False
        trees = []
        if len(tree_file) != 0 and tree_file.find('\x00') != 0:
            trees = ast.literal_eval(tree_file)
            for i in range(3):
                if trees[0][i] != param[i]: clear = True
        else:
            clear = True
        if clear:
            with open("../output/tree.txt","wb") as t:
                trees = [param, createTree(train_cases, header, nparent, nchild, mind, maxdepth)]
                t.write(trees.__repr__())
        return trees[1]
#####################################
# MAIN SCRIPT: Build, Classify, Output
#####################################
# Setup
#input loop for PLV settings
#arguments to script are [pignisitic type(1-4), output comparison type(1-4), output file name(string), testing/traing(y/n)]
args = sys.argv[1:]
print("args: ", args)
#hyperparameters
maxdepth = 10
nparent = 10
nchild = 15


global pign_type

if len(args) == 0:
    pign_type = None
    while pign_type != 1 and pign_type != 2 and pign_type != 3 and pign_type != 4:
        pign_type = int(input("\n\nPignistic Type?\n1.Mean\n2.Median\n3.Mode\n4.Distribution\n\ntype: "))

    pign_type = None
    while pign_type != 1 and pign_type != 2 and pign_type != 3 and pign_type != 4:
        pign_type = input("\n\nOutput Type?\n1.Mean\n2.Median\n3.Mode\n4.Distribution\n\ntype: ")

    # file output settings
    f_name = raw_input("\n\nfile for confusion matrix: ")

    #input loop for variable settings
    var_set = None
    while var_set != "y" and var_set != "n":
        var_set = raw_input("\n\ntesting?(y/n): ")

elif len(args) >= 3:
    pign_type = int(args[0])
    f_name = args[1]
    var_set = args[2]
    if len(args) >= 6:
        print("setting hyperparams")
        maxdepth = int(args[3])
        nparent = int(args[4])
        nchild = int(args[5])
else:
    print("arguments to script are [pignisitic type(1-4), output file name(string), testing/traing(y/n)]")
    sys.exit()

#open the output file
f = open(f_name, "w")

importIdData("../../data/clean/LIDC_809_Complete.csv")

if(var_set == "n"):
    importAllData("../../data/modeBalanced/ModeBalanced_170_LIDC_809_Random.csv")
elif(var_set == "y"):
    importAllData("../../data/modeBalanced/testing_file.csv")

test_header = copy.copy(header)

all_data = LIDC_Data.tolist()

#splitting data
train_cases, test_cases = train_test_split(all_data, test_size=0.3, random_state=42)

global id_list
id_list = []

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

if pign_type == 1 or pign_type == 2 or pign_type == 3:
    train_actual = [int(getActualLabel(x[-4:])) for x in train_cases]
    test_actual = [int(getActualLabel(x[-4:])) for x in test_cases]
elif pign_type == 4:
    train_actual = [getMax(getDist(x[-4:])) for x in train_cases]
    test_actual = [getMax(getDist(x[-4:])) for x in test_cases]

# Console Output
print("#################################")
print("Train Size: ", len(train_cases))
print("Test Size: ", len(test_cases))
print ("Building Belief Decision Tree...")

# Create Tree
# setting "switch = True" will make new tree each time
print("maxdepth: ", maxdepth)
print("nparent: ", nparent)
print("nchild: ", nchild)
tree = getTrees(train_cases, header, nparent, nchild, 0, maxdepth, True)

train_features = [x[:-4] for x in train_cases]
test_features = [x[:-4] for x in test_cases]

# Classify training set
print ("Classifying Training Set...")
for i in range(0,len(train_cases)):
    train_predicted.append(classify(tree, test_header,train_features[i]))

# Classify testing set
print ("Classifying Testing Set...")
for i in range(0,len(test_cases)):
    test_predicted.append(classify(tree, test_header,test_features[i]))

test_pred = test_predicted
# Save training metrics
train_conf_matrix = []
test_conf_matrix = []

#test_predicted = getPredictedLabels(test_predicted)
test_predicted = getPredictedLabels(test_predicted)

print(confusion_matrix(test_actual, test_predicted, [1,2,3,4,5]), file=f)
print(accuracy_score(test_actual,test_predicted), file=f)


test_act = getPigns(test_cases)

correct_count = 0
total_count = len(test_cases)

#for i in range(0, len(test_cases)):
#    correct = "I"
#    if(test_predicted[i] == test_actual[i]):
#        correct = "C"
#        correct_count+=1
#    print(correct, i, ": (", test_actual[i], ",", test_predicted[i], ") -- ",": actual: ", test_act[i], " predicted: ", test_pred[i], file=f)
#
#print("accuracy: ", float(correct_count)/total_count, file=f)


# write training data

# Close output file
f.close()
