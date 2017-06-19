"""""""""""""""""""""
CUSTOM BDT BUILD
Created on Wed Jul 06 13:54:20 2016

@authors: { Rachael Affenit <raffenit@gmail.com>,
            Erik Barns <erik.barns909@gmail.com> }
            
@INPROCEEDINGS { 
  pele2008,
  title={A linear time histogram metric for improved sift matching},
  author={Pele, Ofir and Werman, Michael},
  booktitle={Computer Vision--ECCV 2008},
  pages={495--508},
  year={2008},
  month={October},
  publisher={Springer} }
"""""""""""""""""""""
#####################################
# BELIEF DECISION TREE IMPLEMENTATION
#####################################
# General Imports
from __future__ import print_function
import math # Used for logarithm and sqrt
import scipy # Used for integration
import matplotlib.pyplot as plt # Plotting decision trees
import pandas as pd # Don't delete this. Just don't.
import numpy as np # Numpy arrays
import csv # Read and write csv files
import copy
from sklearn.metrics import confusion_matrix # Assess misclassification
from pyemd import emd # Earth mover's distance
from scipy import spatial #cosine similarity 

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
    
def importData(test_name, train_name):
    global header  
    global tst_data_array
    global trn_data_array
    
    tst_f =  test_name # import file
    tst_df = pd.read_csv(tst_f, header = 0)
    header = list(tst_df.columns.values)[:-4]
    tst_df = tst_df._get_numeric_data()
    tst_data_array = tst_df.as_matrix()
    
    trn_f =  train_name # import file
    trn_df = pd.read_csv(trn_f, header = 0)
    trn_df = trn_df._get_numeric_data()
    trn_data_array = trn_df.as_matrix()
    
# SET TRAINING DATA
def setTrain():
    global train_features
    train_features = trn_data_array[:].tolist() # train pixel area

# SET TESTING DATA
def setTest():
    global test_features
    test_features = tst_data_array[:].tolist() # test pixel area
    
# EXTRACT DATA
def getPigns(dataset):
    nodePigns = []
    # For each case at this node, calculate a vector of probabilities
    for case in dataset:
        currentLabel = case[-4:]
        zeroratings = 0
        pign = []
        
        # Convert radiologist ratings into pignistic probability distributions
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
                gainRatio = float(infoGain) / (-1*splitInfo)
                
                if(gainRatio > bestGain):
                    bestGain = gainRatio 
                    bestFeature = j
                    bestValue = val
                    #print ("Gain = ",infoGain,file=f)
                    #print ("Gain Ratio = ",gainRatio,file=f)
    
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
            print ("Stopping conditions met",file=f)
            print ("LEAF >>>>>>>>>>>\n",file=f)
            decision_tree = {"Leaf":{"BBA":baseApign}}
            return decision_tree
        elif (bestGainRatio == 0):
            print ("Found leaf node with Gain = 0",file=f)
            print ("LEAF >>>>>>>>>>>\n",file=f)
            decision_tree = {"Leaf":{"BBA":baseApign}}
            return decision_tree
        elif (all_same(basePigns)):
            print ("Found leaf node with all BBA's equal",file=f)
            print ("LEAF >>>>>>>>>>>\n",file=f)
            decision_tree = {"Leaf":{"BBA":baseApign}}
            return decision_tree
        elif (curr_depth == max_depth): 
            print ("Max depth reached: ",str(max_depth),file=f)
            print ("LEAF >>>>>>>>>>>\n",file=f)
            decision_tree = {"Leaf":{"BBA":baseApign}}
            return decision_tree
            
        # Recursive Call
        else:
            print ("\tSplitting at depth ", curr_depth, "...")  
            print ("Length of label set = " + str(len(labels)),file=f)
            print ("Selected feature to split = " + str(bestFeatLabel) + ", " + str(bestFeat),file=f)
            print ("Best value = " + str(bestVal),file=f)
            print ("bestGainRatio = " + str(bestGainRatio), file=f)            
            
            # Create a split
            decision_tree[bestFeatLabel][bestVal] = {}            
            subLabels = labels[:]
            lowSet, highSet = splitData(dataset, bestFeat, bestVal) # returns low & high set
            print ("Set Lengths (low/high) = ",str(len(lowSet))," / ",str(len(highSet)),file=f)
            print("SPLIT >>>>>>>>>>>\n", file=f)            
            
            # Recursively create child nodes
            decision_tree[bestFeatLabel][bestVal]["left"] = createTree(lowSet, subLabels, min_parent, min_child,\
                                                                       curr_depth+1, max_depth)
            decision_tree[bestFeatLabel][bestVal]["right"] = createTree(highSet, subLabels, min_parent, min_child,\
                                                                        curr_depth+1, max_depth)
            
            return decision_tree
    
    # If no labels left, then STOP
    elif (len(labels) < 1):
        print ("All features have been used to split; all further nodes are leaves", file=f)
        print("LEAF >>>>>>>>>>>\n",file=f)
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

# Calculate a confusion matrix given predicted and actual
def getConfusionMatrix(predicted, actual): #returns misclassification matrix
    # TEST: Find greatest probability to use as prediction
    pred_proba = [(case.index(max(case))+1) for case in predicted]
    act_proba = [(case.index(max(case))+1) for case in actual]
    return confusion_matrix(act_proba,pred_proba) # Top axis is predicted, side axis is actual

# Calculate accuracy given a confusion matrix
def getAccuracy(class_matrix, num_cases): # returns accuracy of misclassification matrix 
    accy = 0.0
    for j in range(0,5):
        accy += class_matrix[j][j]
    return accy / num_cases

# Get area under ROC curve for a single case
def getROC(predicted, actual):
    #print(predicted," | ",actual)
    binaryActual = [0, 0, 0, 0, 0]
    maxIndex = actual.index(max(actual))
    binaryActual[maxIndex] = 1
    roc = roc_auc_score(binaryActual, predicted)
    return roc
    
# Calculate the cosine similarity between predicted and actual
def CosineSimilarity(actual, predicted):
    similarities= []
    average = 0.0
    for i in range(len(predicted)):
        temp = spatial.distance.cosine(actual[i],predicted[i])
        similarities.append(temp)
        average += temp
    ratings = {0.25:0, 0.5:0, 0.75:0, 1:0}
    for k in ratings.keys():
        ratings[k] = len([x for x in similarities if x <= k and x >= k - 0.25])
        print(ratings)
    return (average/len(predicted), similarities, ratings)
    
#CALCULATE THE EMD OF THE TWO VECTORS   
def My_emd(v1, v2):  # REWROTE THIS FROM AN APACHE SITE, I AM NOT CONFIDENT IN THIS CODE 
    lastDistance = 0
    totalDistance = 0
    for i in range(len(v1)):
        currentDistance = (v1[i] + lastDistance) - v2[i]
        totalDistance += abs(currentDistance)
        lastDistance = currentDistance
    return totalDistance         
# Calculate AUCdt given predicted, actual, and distance threshold
def AUCdt(x, actual, predicted): 
    dist = np.array([[1., 1., 1., 1., 1.],\
                     [1., 1., 1., 1., 1.],\
                     [1., 1., 1., 1., 1.],\
                     [1., 1., 1., 1., 1.],\
                     [1., 1., 1., 1., 1.]]) # THIS IS NOT A REAL DISTANCE MATRIX   
#    dist = np.array([[2., 0., 0., 0., 0.],\
#                     [0., 2., 0., 0., 0.],\
#                     [0., 0., 2., 0., 0.],\
#                     [0., 0., 0., 2., 0.],\
#                     [0., 0., 0., 0., 2.]]) # THIS IS NOT A REAL DISTANCE MATRIX   
    AUCsum = 0.0
    counter = 0
    #if x < .01:
        #print ("EMD for same: ",emd(np.asarray([0.,0.,0.,0.,1.]), np.asarray([0.,0.,0.,0.,1.]), dist), file=f)
        #print ("EMD for 1 off: ",emd(np.asarray([0.,0.,0.,.25,.75]), np.asarray([0.,0.,0.,.5,.5]), dist), file=f)
        #print ("EMD for 1 off: ",emd(np.asarray([0.,0.,.75,.25,0.]), np.asarray([0.,0.,.5,.5,0.]), dist), file=f)
    
    # Sum all EMD values <= x
    for i in range(0, len(predicted)): # NEED TO ADD DISTANCE THRESHOLD OF X HERE

        temp = emd(np.asarray(actual[i]), np.asarray(predicted[i]), dist)
        if  temp <= x:
            AUCsum += temp
            counter += 1
    if counter == 0:
        return AUCsum
    else:
        return AUCsum / counter # Return AUCdt for integration
    
#####################################
# OUTPUT RESULTS
#####################################
def writeData(trainOrTest, header, filename, params, actual, predicted, id_start):
    with open(filename, params) as csvfile:
        avgROC = 0
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Nodule ID',\
                         'Actual [1]',      'Actual [2]',       'Actual [3]',       'Actual [4]',       'Actual [5]',\
                         'Predicted [1]' ,  'Predicted [2]',    'Predicted [3]',    'Predicted [4]',    'Predicted [5]',\
                         'RMSE', 'Pearson r', 'ROC AUC', header])
        
        # For each case at this node    
        for i in range(0, len(predicted)):
            rmse = 0            
            r = 0
            xySum = 0
            xSum = 0
            ySum = 0
            roc = getROC(predicted[i], actual[i])
            avgROC += roc
            
            # Calculate RMSE and Pearson Correlation
            for j in range (0, 5):
                rmse += pow((actual[i][j] - predicted[i][j]), 2)
                xySum += (actual[i][j] - sum(actual[i])/5) * (predicted[i][j] - sum(predicted[i])/5)
                xSum += pow((actual[i][j] - sum(actual[i])/5),2)
                ySum += pow((predicted[i][j] - sum(predicted[i])/5),2)
            rmse /= 5
            rmse = math.sqrt(rmse)
            r = xySum / (math.sqrt(xSum) * math.sqrt(ySum))
            
            # Write data and similarity measures to file
            writer.writerow([set_data_array[i+id_start][2],\
                             actual[i][0], actual[i][1], actual[i][2], actual[i][3], actual[i][4],\
                             predicted[i][0], predicted[i][1], predicted[i][2], predicted[i][3], predicted[i][4],\
                             rmse, r, roc])
            
    # Calculate Accuracy and AUCdt
    conf_matrix = getConfusionMatrix(predicted, actual)
    accy = getAccuracy(conf_matrix, len(predicted))
    myAUCdt = scipy.integrate.quad(AUCdt, 0, 1, args = (actual, predicted))
    myAUCdt = [0,0]
    avgROC /= len(predicted)
    cos_sim, similarities,ratings = CosineSimilarity(actual, predicted)
    
    # Output Confusion Matrix, Accuracy, AUCdt
    print("\n", trainOrTest, " Confusion Matrix", file=f)
    print(conf_matrix,file=f)
    print("\n", trainOrTest, " Accuracy = ", accy, file=f)
    #print(trainOrTest, " AUCdt = ", myAUCdt[0], " with error of ", myAUCdt[1], file=f)
    print("Average Cosine Similarity: ", cos_sim,file=f)
    print("Average ROC AUC Score: ", avgROC, file=f)
    print("Cosine Counts: ", file=f)
    for k in ratings.keys(): 
        print("("+str(k-0.25) + " - " + str(k)+")", ratings[k],file=f)

#####################################
# MAIN SCRIPT: Build, Classify, Output
#####################################

# Setup
f = open("output/BDT Output.txt","w")
treeFile = open("output/tree.txt","w")
trainLabels = []
testLabels = []
importIdData("./data/clean/LIDC_809_Complete.csv")
importData("./data/balanced/Balanced(40)_Clean_809_Test.csv","./data/balanced/Balanced(120)_Clean_809_Train.csv")
setTrain()
setTest()
test_header = copy.copy(header)

# Create Tree
print ("Building Belief Decision Tree...") 
nparent = 27
nchild = 13
maxdepth = 50
print("CREATING BDT (d = ",maxdepth,", np = ",nparent,", nc = ",nchild,"): \n\n",file=f)
tree = createTree(train_features, header, nparent, nchild, 0, maxdepth)

print("BDT TREE: ",file = f)
print(tree, file = f)
print(tree, file = treeFile)
#createPlot(tree)

# Get actual data
actualTrain = getPigns(train_features)
actualTest = getPigns(test_features)

# Classify training and testing sets
print ("Classifying Training Set...") 
for i in range(0,len(train_features)):
        trainLabels.append(classify(tree, test_header,train_features[i]))
print ("Classifying Testing Set...") 
for i in range(0,len(test_features)):
        testLabels.append(classify(tree, test_header,test_features[i]))

# Output data to csv and text files
print ("Writing Data...") 
trainhead = "Training Data: (d = " + str(maxdepth) + " | np = " + str(nparent) + " | nc = " + str(nchild) + ")"
testhead = "Testing Data: (d = " + str(maxdepth) + " | np = " + str(nparent) + " | nc = " + str(nchild) + ")"
writeData("Training ", trainhead, "output/TrainOutput.csv", "wb", actualTrain, trainLabels, 0)
writeData("Testing ", testhead, "output/TestOutput.csv", "wb", actualTest, testLabels, len(trainLabels))

# Close output file
f.close()

print ("DONE") 
