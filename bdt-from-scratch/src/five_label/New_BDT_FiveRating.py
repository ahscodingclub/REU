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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv # Read and write csv files
import time # for testing purposes
import copy
from sklearn.metrics import confusion_matrix # Assess misclassification
from scipy import spatial # Cosine similarity 
import ast
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

        if pignType == 1:      #mean
            sum = 0
            for i in currentLabel:     # Count number of instances of each rating
                sum += i
            mean = sum/4 
            pign[int(math.floor(mean))-1] = 1 - (mean - math.floor(mean))
            if mean-int(math.floor(mean)) != 0:
                pign[int(math.floor(mean))] = mean - math.floor(mean)

        elif pignType == 2:    #median
            a = np.array(currentLabel)
            median = np.median(a)
            if float(median).is_integer():
                pign[int(math.floor(median))-1] = 1
            else:
                pign[int(math.floor(median))-1] = 1 - (median - math.floor(median))
                if median-math.floor(median) != 0:
                    pign[int(math.floor(median))] = median - math.floor(median)

        elif pignType == 3:    #mode (which appears most often, majority vote)
            mode = int(scipy.stats.mode(np.array(currentLabel))[0][0])
            pign[mode-1] = 1

        elif pignType == 4:    #distribution
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
def getConfusionMatrix(predicted,actual):
  return cross_product(predicted,actual)

def cross_product(X,Y):
    product = [ [ 0 for y in range(len(Y)) ] for x in range(len(X)) ]
    # iterate through rows of X
    for i in range(len(X)):
        # iterate through rows of Y
        for j in range(len(Y)):
            product[i][j] += X[i] * Y[j]
    return product

# Calculate accuracy given a confusion matrix
# returns accuracy of misclassification matrix 
def getAccuracy(class_matrix): 
    accy = 0.0
    for j in range(0,5):
        accy += class_matrix[j][j]
    return 100 *accy / sum2D(class_matrix)

# Calculate the maximum accuracy that can be achieved from a
# probabilistic label vector
def getCredibility(plv):
    return getAccuracy(cross_product(plv, plv))

# Sum elements in a 2D array
def sum2D(input):
    return sum(map(sum, input))

def getPAActual(predicted):
  actual = [0,0,0,0,0] * len(predicted)
  csv_f = open(f_name[0:-4] + '_training' + '.csv')
  csv_f = csv.reader(csv_f)
  csv_f.next()

  for i in range(0, len(predicted)):
    pred = predicted[i]
    find_count = 0
    for row in csv_f: 
      train_act = [row[1], row[2], row[3], row[4], row[5]]
      train_act = [float(x) for x in train_act]
      train_pred = [row[6], row[7], row[8], row[9], row[10]]
      train_pred = [float(x) for x in train_pred]
      if(pred == train_pred):
        actual[i] = np.add(actual[i], train_act)
        find_count+=1
    actual[i] = np.divide(actual[i], find_count)

  return actual

# Calculate Jeffreys Distance of two vectors
def JeffreyDistance(v1,v2):
    out = 0
    for i in range(len(v1)):
        m = (v2[i] + v1[i])/2
        if m != 0:
            
            if v1[i] == 0: 
                a = 0
            else:
                a = v1[i] * math.log(v1[i]/m)
            if v2[i] == 0:
                b = 0
            else:
                b = v2[i] * math.log(v2[i]/m)
            out += (a + b) 
    return out
 
#####################################
# OUTPUT RESULTS DATAFILES
#####################################
def writeData(train_or_test, filename, actual, predicted, confusion, credibility, confidence, id_start,training):
    
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if train_or_test == "Training":
            writer.writerow(['Nodule ID',\
                             'Actual [1]',      'Actual [2]',       'Actual [3]',       'Actual [4]',       'Actual [5]',\
                             'Predicted [1]' ,  'Predicted [2]',    'Predicted [3]',    'Predicted [4]',    'Predicted [5]',\
                              header])
        else:
            writer.writerow(['Nodule ID',\
                             'Actual [1]',      'Actual [2]',       'Actual [3]',       'Actual [4]',       'Actual [5]',\
                             'Predicted [1]' ,  'Predicted [2]',    'Predicted [3]',    'Predicted [4]',    'Predicted [5]',\
                             'Confidence', 'Credibility', header])

        
        for i in range(0, len(predicted)):
            # Write data and similarity measures to file
            if train_or_test == "Training":
                writer.writerow([set_data_array[i+id_start][2],\
                                 actual[i][0], actual[i][1], actual[i][2], actual[i][3], actual[i][4],\
                                 predicted[i][0], predicted[i][1], predicted[i][2], predicted[i][3], predicted[i][4],\
                                ])
            else:
                writer.writerow([set_data_array[i+id_start][2],\
                                 actual[i][0], actual[i][1], actual[i][2], actual[i][3], actual[i][4],\
                                 predicted[i][0], predicted[i][1], predicted[i][2], predicted[i][3], predicted[i][4],\
                                 confidence[i], credibility[i],])
    if(training):
      return 
    # Computing aggregate confidence and credibility
    agg_conf_matrix = np.multiply(0, copy.deepcopy(confusion[0]))
    agg_confidence = 0
    agg_credibility = 0

    print("\n\n" + train_or_test, file=f)
    for i in range(0,len(confusion)):
        agg_conf_matrix = np.add(agg_conf_matrix, confusion[i])
        agg_credibility += credibility[i]
    
    agg_credibility = agg_credibility/len(credibility)
    agg_confidence = 100*getAccuracy(agg_conf_matrix)/agg_credibility

    print("\n\nAggregate Results: ", file=f)
    for row in agg_conf_matrix:
            print(["{0:5.5}".format(str(val)) for val in row], file=f)

    print("Credibility = ", '{:.4}'.format(float(agg_credibility)), file=f)       
 
    print("Confidence = ", '{:.4}'.format(float(agg_confidence)), file=f)

         
#####################################
# DRAW THE DECISION TREE
#####################################

# IF tree in file is same tree we want, USE IT rather than building it
def getTrees(tf,head,np,nc,mind,md, switch):
    param = [np,nc,md]
    if(switch):
         with open("../output/tree.txt","wb") as t:
            trees = [param,createTree(train_features, header, nparent, nchild, mind, maxdepth)]
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
                trees = [param, createTree(train_features, header, nparent, nchild, mind, maxdepth)]
                t.write(trees.__repr__())
        return trees[1]
    
def draw(parent_name, child_name):
    edge = pydot.Edge(str(parent_name), str(child_name))
    graph.add_edge(edge)

def visit(node, parent=None):
    print (node)
    for k,v in node.iteritems():
        if isinstance(v, dict) and k != 'BBA' and ~isinstance(k, float):
            # We start with the root node whose parent is None
            # we don't want to graph the None node
            if parent:
                draw(parent, k)
            visit(v, k)
        elif isinstance(k, float):
            print ("Split found at value ",v)
            return
        elif  k == 'BBA':
            print ("BBA found: ",v)
            return
        else:
            draw(parent, k)
            # drawing the label using a distinct name
            draw(k, k+'_'+v)

# PLOT EACH VIOLIN PLOT
def plotVio(old, category, a, axes, xlabel, ylabel, title):
    new = []
    labels = []

    for cat in category:
        if (cat not in labels):
            new.append([])
            labels.append(cat)
    labels = sorted(labels)
    
    # Sort category values into new by label
    for x in range (len(old)):
        new[labels.index(category[x])].append(old[x])
        
    # Plot the information
    nans = np.array([float('nan'), float('nan')])
    try:  
        axes[a].violinplot([val or nans for val in new], showmeans=True, showmedians=False)
    except ValueError as e:  #raised if `y` is empty.
        print (e)
        pass
    
    axes[a].set_title(title)
    axes[a].yaxis.grid(True)
    if(title[:9] != "Aggregate"):
        axes[a].set_xticks([y+1 for y in range(0,len(new))])
        axes[a].set_xticklabels(labels)    
    else: 
        axes[a].tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off') 
    axes[a].set_xlabel(xlabel)
    axes[a].set_ylabel(ylabel)
    axes[a].set_ylim([0,102])

# DRAW VIOLIN PLOTS FOR CONFIDENCE / CREDIBILITY
def violin(credibility, confidence, category):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # Confidence
    plotVio(confidence, category, 0, axes,  'Classification of Malignancy', 'Confidence', 'Confidence Values at Each Classification')
    
    # Credibility
    plotVio(credibility, category, 1, axes, 'Classification of Malignancy', 'Credibility', 'Credibilty Values at Each Classification')
    
    # Save figure
    figure_dest = f_name[0:-4] + '_classes.png'
    plt.savefig(figure_dest, bbox_inches='tight')
    print("Figure saved in ", figure_dest)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # Confidence
    plotVio(confidence, [1]*len(category), 0, axes,  'Density of Values', 'Confidence', 'Aggregate Confidence Values of Classifications')
    
    # Credibility
    plotVio(credibility, [1]*len(category), 1, axes, 'Density of Values', 'Credibility', 'Aggregate Credibility Values of Classifications')
    # Save figure

    figure_dest = f_name[0:-4]+'_aggregate.png'
    plt.savefig(figure_dest, bbox_inches='tight')
    print("Figure saved in ", figure_dest)

    
#####################################
# MAIN SCRIPT: Build, Classify, Output
#####################################       
# Setup
#input loop for PLV settings
pignType = None
while pignType != 1 and pignType != 2 and pignType != 3 and pignType != 4:
    pignType = input("Pignistic Type?\n1.Mean\n2.Median\n3.Mode\n4.Distribution\n\ntype: ")

#file output settings
f_name = raw_input("file for confusion matrix: ") 
f = open(f_name, "w")

#input loop for variable settings
var_set = None
while var_set != "y" and var_set != "n":
    var_set = raw_input("testing?(y/n): ")

importIdData("../data/clean/LIDC_809_Complete.csv")

kfolds = 6
nparent = 24
nchild = 12
maxdepth = 25

if(var_set == "n"):
    importAllData("../data/modeBalanced/ModeBalanced_170_LIDC_809_Random.csv")
elif(var_set == "y"):
    importAllData("../data/modeBalanced/testing_file.csv")
    
test_header = copy.copy(header)

###### K-FOLD VALIDATION ######
kf = KFold(len(LIDC_Data), kfolds)
    
k_round = 1
k_best = [None]*7

trainhead = "Training Data: (d = " + str(maxdepth) + " | np = " + str(nparent) + " | nc = " + str(nchild) + " | k = " + str(kfolds) + ")"

testhead = "Testing Data: (d = " + str(maxdepth) + " | np = " + str(nparent) + " | nc = " + str(nchild) + " | k = " + str(kfolds) + ")"

print("Classifying with BDT Parameters (d = ",maxdepth,", np = ",nparent,", nc = ",nchild,", k = ",kfolds,"):\n",file=f)

global graph

for trn_ind, tst_ind in kf:
    trainLabels = []
    testLabels = []
    setTrain(LIDC_Data[trn_ind])
    test_features = LIDC_Data[tst_ind].tolist()

    # Get actual data
    actualTrain = getPigns(train_features)
    actualTest = getPigns(test_features)
    
    # Console Output
    print("\n K-FOLD VALIDATION ROUND ",k_round," OF ",kfolds)
    print("#################################")
    print("Train Size: ", len(train_features))
    print("Test Size: ", len(test_features))
    print ("Building Belief Decision Tree...") 
    
    # Create Tree
    # setting "switch = True" will make new tree each time
    tree = getTrees(train_features, header, nparent, nchild, 0, maxdepth, True) 

    #graphing the tree
#    graph = pydot.Dot(graph_type='graph')
#    visit(tree)
#    graph.write_png("BDT.png")
    
    # Classify training set
    print ("Classifying Training Set...") 
    for i in range(0,len(train_features)):
            trainLabels.append(classify(tree, test_header,train_features[i]))

   # Classify testing set
    print ("Classifying Testing Set...") 
    for i in range(0,len(test_features)):
            testLabels.append(classify(tree, test_header,test_features[i]))


    classes = [testlabel.index(max(testlabel))+1 for testlabel in testLabels] 

    # Save training metrics
    train_conf_matrix = []
    train_credibility = []
    train_confidence = []

    # Save testing metrics
    test_conf_matrix = []
    test_credibility = []
    test_confidence = []
    
    # Calculate the confusion matrix, accuracy, credibility and confidence 
    #   of each training case 
    for i in range(len(trainLabels)):
        train_conf_matrix.append(getConfusionMatrix(trainLabels[i], actualTrain[i]))
        train_credibility.append(getCredibility(actualTrain[i]))
        accy = 100*getAccuracy(train_conf_matrix[i])
        if accy > train_credibility[i]:
            accy = train_credibility[i] - (accy - train_credibility[i])
        train_confidence.append(accy/train_credibility[i])

    training_data = [train_conf_matrix, train_credibility, train_confidence, classes]
  
    writeData("Training", f_name[0:-4] + '_training' + '.csv', actualTrain, trainLabels, training_data[0], training_data[1], training_data[2], 0, True)
    
    ## P->A Heuristic (predicted to actual mapping for testing set)
    pa_heur_act_test = getPAActual(testLabels)
    
    # Calculate the confusion matrix, accuracy, credibility and confidence 
    #   of each testing case   
    for i in range(len(testLabels)):
        test_conf_matrix.append(getConfusionMatrix(testLabels[i], actualTest[i]))
        test_credibility.append(getCredibility(actualTest[i]))
        accy = 100*getAccuracy(test_conf_matrix[i])
        if accy > test_credibility[i]:
            accy = test_credibility[i] - (accy - test_credibility[i])
        test_confidence.append(accy/test_credibility[i])

    testing_data = [test_conf_matrix, test_credibility, test_confidence, classes]

    accuracy = np.sum(test_confidence)/len(test_confidence)
    
    print("accuracy: ", accuracy, " for fold ", k_round)
   
    if accuracy > k_best[1]:
        k_best = [k_round, accuracy, actualTrain, trainLabels, actualTest, testLabels, training_data, testing_data]

    # increase round of k-fold vlaidation
    k_round += 1

# Output data to csv and text files
actualTrain = k_best[2]
trainLabels = k_best[3]
actualTest = k_best[4]
testLabels = k_best[5]
training_data = k_best[6]
testing_data = k_best[7]

print ("\nWriting Data for best fold k =", k_best[0], "...\n") 

# write training data
writeData("Training", f_name[0:-4] + '_training' + '.csv', actualTrain, trainLabels, training_data[0], training_data[1], training_data[2], 0, False)

# write testing data
writeData("Testing", f_name[0:-4] + '_testing' + '.csv', actualTest, testLabels, testing_data[0], testing_data[1], testing_data[2], len(trainLabels), False)

# plot violin plots
violin(testing_data[1],testing_data[2], testing_data[3])
    
# Close output file
f.close()
