"""
CUSTOM BDT BUILD
Created on Wed Jul 06 13:54:20 2016

@authors:   Rachael Affenit <raffenit@gmail.com>
            Erik Barns <erik.barns909@gmail.com>
"""
#####################################
# BELIEF DECISION TREE IMPLEMENTATION
#####################################
# General Imports
from __future__ import print_function
import math
import numpy as np
import csv
from math import log
import copy
import pandas as pd
import ast
from sklearn.metrics import confusion_matrix # Assess misclassification
from sklearn.metrics import roc_auc_score # Get ROC area under curve
from scipy import spatial #cosine similarity 
import scipy
from sklearn.cross_validation import KFold
#####################################
# CREATE TREE 
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
        
def importData(test_name, train_name, calib_name):
    global header  
    global tst_data_array
    global trn_data_array
#    global cal_data_array
    
    tst_f =  test_name # import file
    tst_df = pd.read_csv(tst_f, header = 0)
    header = list(tst_df.columns.values)[:-4]
    tst_df = tst_df._get_numeric_data()
    tst_data_array = tst_df.as_matrix()
    
    trn_f =  train_name # import file
    trn_df = pd.read_csv(trn_f, header = 0)
    trn_df = trn_df._get_numeric_data()
    trn_data_array = trn_df.as_matrix()
    
#    cal_f =  calib_name # import file
#    cal_df = pd.read_csv(cal_f, header = 0)
#    header = list(cal_df.columns.values)[:-4]
#    cal_df = cal_df._get_numeric_data()
#    cal_data_array = cal_df.as_matrix()
    
# SET TRAINING DATA
def setTrain(lst):
    global train_features
    global calib_features
    #rand_trn = [ trn_data_array[i].tolist() for i in random.sample(xrange(len(trn_data_array)), len(trn_data_array)) ]
    #np.random.shuffle(trn_data_array)
    split = int((6.0/7.0) * len(lst))
    #print(split)
    train_features = lst[:split].tolist() # training data
    #calib_features = [ train_features[i] for i in sorted(random.sample(xrange(len(train_features)), split)) ]
    calib_features = lst[split:].tolist() # calibration data
    
# EXTRACT DATA
def getPigns(dataset, ratings):
    nodePigns = []
    # For each case at this node, calculate a vector of probabilities
    for case in dataset:
        if ratings != 4:
            currentLabel = case[-4:-4+ratings]
        else:
            currentLabel = case[-4:]
        
        zeroratings = 0
        pign = []
        
        # Convert radiologist ratings into pignistic probability distributions
        for i in range (0, 6): # Count number of instances of each rating
            if i == 0:
                zeroratings += currentLabel.count(i)
            else:
                pign.append(currentLabel.count(i))
        
        pign[:] = [float(x) / (ratings - zeroratings) for x in pign] 
#        pign = five_to_two(pign)
        nodePigns.append(pign) # Add pign to list of pigns
    return nodePigns
    
def five_to_two(lst):
    theta = lst[2]/2
    pb = lst[0] + (lst[1] * 0.75) + theta + (lst[3] * 0.25)
    pm = lst[4] + (lst[3] * 0.75) + theta + (lst[1] * 0.25)
    return [pb,pm]
    
# SET TESTING DATA
#def setTest():
#    global test_malig    
#    global test_features
#    global tstm    
#    test_malig = []
#    #test_malig = tst_data_array[:,-1] # test malignancy
#    test_features = tst_data_array[:] # test pixel area
#    test_features = test_features.tolist()
#    tstm = [int(x) for x in test_malig]

# CALCULATE ENTROPY OF A NODE
def calcEntrop(dataset,num_ratings):
    nodePigns = getPigns(dataset,num_ratings)
    
    # Compute average pignistic probability
    Apign = [0.0, 0.0, 0.0,0.0,0.0]
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
    
    for case in dataset:
        if case[splitFeat] < splitValue:
            lowSet.append(case)
        elif case[splitFeat] >= splitValue:
            highSet.append(case)
        
    return lowSet, highSet

# DETERMINE BEST FEATURE
def bestFeatureSplit(dataset, nFeat, min_parent, min_child,num_ratings):
    numFeat = nFeat # 65 features
    baseEntrop, baseApign, basePigns = calcEntrop(dataset,num_ratings) # Entropy of main node
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
    
                    entrop = calcEntrop(subDataset[i],num_ratings)[0]
                    
                    newEntrop += prob * entrop # Sum over both datasets
                    splitInfo += prob * log(prob2, 2)
                        
                # Calculate Gain Ratio
                infoGain = baseEntrop - newEntrop
                if splitInfo == 0:
                    pass
                else:
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

def createTree(dataset, labels, min_parent, min_child, curr_depth, max_depth, num_ratings):
    # If labels exist in this subset, determine best split
    if len(labels) >= 1:
        
        # Get Best Split Information
        output = bestFeatureSplit(dataset, len(labels), min_parent, min_child,num_ratings) # index of best feature
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
                                                                       curr_depth+1, max_depth,num_ratings)
            decision_tree[bestFeatLabel][bestVal]["right"] = createTree(highSet, subLabels, min_parent, min_child,\
                                                                        curr_depth+1, max_depth,num_ratings)
            
            return decision_tree
    
    # If no labels left, then STOP
    elif (len(labels) < 1):
        print ("All features have been used to split; all further nodes are leaves", file=f)
        print("LEAF >>>>>>>>>>>\n",file=f)
        return
#####################################
# OUTPUT RESULTS DATAFILES
#####################################
def writeData(trainOrTest, header, filename, params, actual, predicted, conf, cred, id_start):
    avgROC = 0    
    rocList = []
    with open(filename, params) as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if trainOrTest == "Training":
            writer.writerow([
                             'Actual [1]',      'Actual [2]',       'Actual [3]',       'Actual [4]',       'Actual [5]',\
                             'Predicted [1]' ,  'Predicted [2]',    'Predicted [3]',    'Predicted [4]',    'Predicted [5]',\
                             'RMSE', 'Pearson r', 'ROC AUC', header])
        else:
            writer.writerow([
                             'Actual [1]',      'Actual [2]',       'Actual [3]',       'Actual [4]',       'Actual [5]',\
                             'Predicted [1]' ,  'Predicted [2]',    'Predicted [3]',    'Predicted [4]',    'Predicted [5]',\
                             'RMSE', 'Pearson r', 'ROC AUC', 'Confidence', 'Credibility', 'Classes', header])
        
        # For each case at this node
        #print(len(actual),len(predicted))
        for i in range(0, len(predicted)):
            rmse = 0            
            r = 0
            xySum = 0
            xSum = 0
            ySum = 0
            roc = getROC(predicted[i], actual[i])
            avgROC += roc
            rocList.append(roc)
            
            # Calculate RMSE and Pearson Correlation
            for j in range (0, len(actual[i])):
                rmse += pow((actual[i][j] - predicted[i][j]), 2)
                xySum += (actual[i][j] - sum(actual[i])/5) * (predicted[i][j] - sum(predicted[i])/5)
                xSum += pow((actual[i][j] - sum(actual[i])/5),2)
                ySum += pow((predicted[i][j] - sum(predicted[i])/5),2)
            rmse /= len(actual[i])
            rmse = math.sqrt(rmse)
            r = xySum / (math.sqrt(xSum) * math.sqrt(ySum))
            
            # Write data and similarity measures to file
            if trainOrTest == "Training":
                writer.writerow([
                                 actual[i][0], actual[i][1], actual[i][2], actual[i][3], actual[i][4],\
                                 predicted[i][0], predicted[i][1], predicted[i][2], predicted[i][3], predicted[i][4],\
                                 rmse, r, roc])
            else:
                writer.writerow([
                                 actual[i][0], actual[i][1], actual[i][2], actual[i][3], actual[i][4],\
                                 predicted[i][0], predicted[i][1], predicted[i][2], predicted[i][3], predicted[i][4],\
                                 rmse, r, roc, conf[i], cred[i]])
                                 
#####################################
# CONFORMAL PREDICTION
#####################################
# Sum elements in a 2D array
def sum2D(input):
    return sum(map(sum, input))

# MODIFIED ACCURACY EQUATION
# MODIFIED ACCURACY EQUATION
def modConfusion(pred, actual):
    matrix = [[0,0,0,0,0],
              [0,0,0,0,0],
              [0,0,0,0,0],
              [0,0,0,0,0],
              [0,0,0,0,0]]
    
    # Build matrix
    for i in range (0,len(pred)):
        dot = [[0,0,0,0,0],
               [0,0,0,0,0],
               [0,0,0,0,0],
               [0,0,0,0,0],
               [0,0,0,0,0]]
        # Multiply elements of actual and predicted together to form a 5x5 matrix
        for j in range (0,5):
            for k in range (0,5):
                dot[j][k] = (float(actual[i][j]) * float(pred[i][k]))
        
        # Add said matrix to final matrix
        for j in range (0,5):
            for k in range (0,5):
                temp = matrix[j][k]
                matrix[j][k] = temp + dot[j][k]
                
    # Normalize matrix
    for j in range (0,5):
        for k in range (0,5):
            matrix[j][k] /= len(pred)
                
    #print (matrix)
    return matrix
    
def getMax(lst):
        mx = max(lst)
        return lst.index(mx)
    
def Train_Conformity(actual, predicted):
    conform = []
    for i in range(len(actual)):
        #actual_class = getMax(actual[i]) #maximum value of predicted
        max_ind = getMax(actual[i]) #index of max actual
        #max_ind = actual_class        
        pred = predicted[i][max_ind] #probability of predicted chosen predicted label
        max_val = max([prob for k,prob in enumerate(predicted[i]) if k != max_ind]) #max value in subset without chosen label
        conform.append(pred - max_val) #append conformity score
    return conform

def Test_Conformity(predicted):
    conform = []
    for i in range(len(predicted)): #for each case
        conf_score = []
        for j in range(len(predicted[i])): #for each rating in case
            cls = predicted[i][j]  #chosen class label
            max_val = max([prob for k,prob in enumerate(predicted[i]) if k != j])#max val of remaining class labels
            conf_score.append(cls-max_val) #chosen - max remaining value
        conform.append(conf_score) #append conformity score
    return conform
    
def PValue(test, calib):
    pvalues = []
    for i in range(len(test)): # for each case in testing 
        counts = [0,0,0,0,0] #stores count of test conformity greater than equal to calib 
        for j in range(5): #for each label in test
            val = test[i][j] #conformity value of label
            for k in range(len(calib)): #for each conformity value 
                if calib[k] <= val: #if less than val increase count of label
                    counts[j] += 1
        pvalues.append([count/float(len(calib)+1) for count in counts]) #append p values for case
    return pvalues
#####################################
# CONFUSION MATRIX BUILDING
#####################################
def getConfusionMatrix(probabilities,actual): #returns misclassification matrix
    # TEST: Find greatest probability to use as prediction
    tst_proba = [getMax(case)+1 for case in probabilities]
    act_proba = [getMax(case)+1 for case in actual]
    return confusion_matrix(act_proba,tst_proba)

def getAccuracy(class_matrix,num_cases): # returns accuracy of misclassification matrix 
    accy = 0.0
    for j in range(5):
        accy += class_matrix[j][j]
            
    return accy / num_cases

def highestConfidence(lst):
    best_confidence = 0.0
    best_vec = []
    for i, val in enumerate(lst):
        m = getMax(val)
        conf = 1 - max([x for k,x in enumerate(val) if k != m])
        if conf >= .875 and i != 0:
            return i
        if conf > best_confidence:
            best_confidence = conf
            best_vec = i
    return best_vec
      

def addWeight(lst,weight):
    for i in range(len(lst)):
        lst[i] = lst[i] * weight
    return lst
    
    
def getConfidentBBA(train_label_dict, calib):
    modified_train = []
    used_trees = [0,0,0,0]
    for j in range(0,len(train_label_dict[1])): #computes the average bba values across all four models 
        bbas = []
        for i in range(1,5): #4 model bbas for one case
                bbas.append(train_label_dict[i][j])
                
        conform = Test_Conformity(bbas)
        p_values = PValue(conform, calib)  
        temp = highestConfidence(p_values) 
        used_trees[temp] += 1
        choice = bbas[temp]
        modified_train.append(choice)
    print("\nDistribution of trees used: ", used_trees)
    return modified_train    
    
def getAverageBBA(train_label_dict):
    modified_train = []
    for j in range(0,len(train_label_dict[1])): #computes the average bba values across all four models 
        avg_bba = [0.0,0.0,0.0,0.0,0.0]
        for i in range(1,5):            
            ####CODE FOR AVERAGE BBA VALUES 
            for k in range(5):
                avg_bba[k] += train_label_dict[i][j][k]
        for i in range(len(avg_bba)):
            avg_bba[i] = avg_bba[i]/4.0
        modified_train.append(avg_bba)
    return modified_train
    
def getMaxBBA(train_label_dict):
    modified_train = []
    dist = [0,0,0,0,0]
    l = 0
    for j in range(0,len(train_label_dict[1])): #computes the average bba values across all four models 
        max_prob = 0
        avg_bba = [0.0,0.0,0.0,0.0,0.0]
        for i in range(1,5):
            if max(train_label_dict[i][j]) >= .50 and i != 1:
                avg_bba = train_label_dict[i][j]
                l = i
                break
            elif max(train_label_dict[i][j]) >= .65:
                avg_bba = train_label_dict[i][j]
                l = i
                break
            elif(max(train_label_dict[i][j]) > max_prob):
                max_prob = max(train_label_dict[i][j])
                avg_bba = train_label_dict[i][j]
                l = i
        dist[l] += 1
        modified_train.append(avg_bba)
    print ("max: ",dist)
    return modified_train        
    
def buildTrees(parent, child, depth):
    trees = []
    parents = [16,16,18,20]
    children = [8,8,9,10]
    
    for i in range(1,5): #create trees with iterative ratings model
        tree = createTree(train_features, header, parents[i-1], children[i-1], 0, depth,i)
        trees.append(tree)
    return trees        

def classifyIterativeTrees(trees,data_header,train_features):
    train_label_dct = {}
    trainLabels = []
    for k in range(1,5): #builds dictionary of bba labels for all four models 
        trainLabels = []
        for i in range(0,len(train_features)):
            trainLabels.append(classify(trees[k-1], data_header,train_features[i]))
        if k in train_label_dct:
            train_label_dct[k].update(trainLabels)
        else:
            train_label_dct[k] = trainLabels
            
    return train_label_dct, trainLabels
    
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
        #print(ratings)
    return (average/len(predicted), similarities, ratings)

# Area Under ROC Curve for a single case
# predicted = [0, .2, .4, .4, 0]
# actual = [0, .25, .5, .25, 0]
def getROC(predicted, actual):
    binaryActual = [0, 0,0,0,0]
    maxIndex = actual.index(max(actual))
    binaryActual[maxIndex] = 1
    roc = roc_auc_score(binaryActual, predicted)
    return roc
  
# Calculate Jeffreys Distance of two vectors
def JeffreyDistance(v1, v2):
    output = 0
    for i in range(len(v1)):
        m = (v2[i] + v1[i])/2
        if m != 0 and v1[i] != 0 and v2[i] != 0:
            output += (v1[i] * math.log((v1[i]/m)))+ v2[i]
            
    return output
    
def JeffreyDistance2(v1,v2):
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
# CALCULATE THE EMD OF THE TWO VECTORS   
def My_emd(v1, v2):  # REWROTE THIS FROM AN APACHE SITE, I AM NOT CONFIDENT IN THIS CODE 
    lastDistance = 0
    totalDistance = 0
    for i in range(len(v1)):
        currentDistance = (v1[i] + lastDistance) - v2[i]
        totalDistance += abs(currentDistance)
        lastDistance = currentDistance
    return totalDistance       

def EMD2(v1 , v2):
      return sum([x for x in abs(np.cumsum(v1) - np.cumsum(v2))])
      
# Calculate AUCdt given predicted, actual, and distance threshold
def AUCdt(x, actual, predicted):  
    AUCsum = 0.0
    counter = 0
    
    # Sum all JD values <= x
    for i in range(0, len(predicted)):
        temp = JeffreyDistance2(np.asarray(actual[i]), np.asarray(predicted[i]))
        #temp = EMD2(np.asarray(actual[i]), np.asarray(predicted[i]))
        if  temp <= x:
            AUCsum += temp
            counter += 1
    if counter == 0:
        return AUCsum
    else:
        return AUCsum / counter # Return AUCdt for integration
        
def getMaxy(lst):
    result = []
    for i in range(len(lst[0])):
        maxy = [0.0,0.0,0.0,0.0,0.0]
        max_val = 0
        for k in range(len(lst)):
            temp = max(lst[k][i])
            if temp > max_val:
                max_val = temp
                maxy = lst[k][i]
        result.append(maxy)
    return result
def getAvy(lst):
    result = []
    for i in range(len(lst[0])):
        subavg = [0.0,0.0,0.0,0.0,0.0]
        for k in range(len(lst)):
            for j in range(len(lst[0][0])):
                subavg[j] += lst[k][i][j]
        subavg = [y/3 for y in subavg]
        result.append(subavg)
    return result
def getMode(lst):
    result = []
    for i in range(len(lst[0])):
        x = [lst[k][i].index(max(lst[k][i]))+1 for k in range(len(lst))]
#        x = [lst[0][i].index(max(lst[0][i])),lst[1][i].index(max(lst[1][i])),lst[2][i].index(max(lst[2][i]))]
        ones = x.count(1)
        twos = x.count(2)
        if (ones > twos):
            result.append(1)
        else:
            result.append(2)
    return result
 
def normalizeProbabilities(lst):
    for item in lst:
        div = sum(item)
        for i in range(len(item)):
            item[i] = item[i]/div
    return lst
#####################################
# CLASSIFY NEW CASES
#####################################
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    
    if (firstStr == 'Leaf'): #IF LEAF NODE, RETURN BBA VECTOR
        bba = inputTree['Leaf'].values()[0]
        return bba
        
    elif (firstStr == 'right' or firstStr == 'left'): #IF RIGHT OR LEFT, RECURSIVE CALL ON SUBTREE OF THAT NODE
        return classify(inputTree[firstStr],featLabels,testVec)
        
    else: #key is integer or feature label
        secondDict = inputTree[firstStr] 
        keys = secondDict.keys()
        featIndex = featLabels.index(firstStr)
        if (keys[0] == 'BBA'): # IF key is BBA, CLASSIFY NEXT KEY IN TREE
            key = keys[1]
        else:
            key = keys[0] #KEY IS SPLIT VALUE 
            
        if testVec[featIndex] > key: #IF GREATER THAN GET SUBTREE RIGHT ELSE GET SUBTREE LEFT
            k = secondDict[key].keys()[0]
            return classify(secondDict[key][k],featLabels,testVec)#GO RIGHT
        else: #GET SUBTREE 
            k = secondDict[key].keys()[1]
            return classify(secondDict[key][k],featLabels,testVec) #GO LEFT

def getTrees(np,nc,md):
    param = [np,nc,md]
    with open("../output/tree.txt","r") as t:
        tree_file = t.read()
    trees = []
    clear = False
    if(len(tree_file)) != 0:
        trees = ast.literal_eval(tree_file)
        for i in range(3): 
            if trees[0][i] != param[i]: clear = True
    else:
        clear = True
    if clear:
         with open("../output/tree.txt","wb") as t:
            trees.append(param)
            trees[1:] = buildTrees(np,nc,md)
            t.write(str(trees))
            t.close()
    return trees[1:]
#####################################
# MAIN SCRIPT
#####################################
f = open("../output/BDT Output.txt","w")
#treeFile = open("output/trees.txt","w")
importAllData("../data/modeBalanced/ModeBalanced_170_LIDC_809_Random.csv")
#importIdData("./data/clean/LIDC_809_Complete.csv")
#importData("./data/modeBalanced/small/Test_ModeBalanced_809.csv","./data/modeBalanced/small/Train_600_ModeBalanced_809_Random.csv","")

kfolds = 5
nparent = 16
nchild = 8
maxdepth = 12

k_round = 1
k_best = [0,0.0,[],[],[],[],[],[]]
kf = KFold(len(LIDC_Data), kfolds)
test_header = copy.copy(header)
i = 0

for trn_ind, tst_ind in kf:
    trnLabels = []
    testLabels = []
    calibLabels = []
    setTrain(LIDC_Data[trn_ind])
    print(len(train_features))
    test_features = LIDC_Data[tst_ind].tolist()
    # Console Output
    print("\n K-FOLD VALIDATION ROUND ",k_round," OF ",kfolds)
    print("#################################")
    print("Train Size: ", len(train_features))
    print("Test Size: ", len(test_features))
    # Create Tree
    print ("Building Belief Decision Tree...") 
    
    
    print("CREATING BDT (d = ",maxdepth,", np = ",nparent,", nc = ",nchild,"): \n\n",file=f)
    # Create Tree
    trees = getTrees(nparent,nchild,maxdepth)
    #trees = buildTrees(nparent,nchild,maxdepth)
    #t.write(str(trees))   
    #trees = [{'MeanIntensityBG': {268.7917: {'right': {'gaborSD_0_0': {1631.7: {'right': {'markov1': {492.4872: {'right': {'gabormean_1_2': {'BBA': [0.08333333333333333, 0.03787878787878788, 0.045454545454545456, 0.030303030303030304, 0.05303030303030303], 113.9333: {'right': {'gabormean_0_2': {4650.11: {'right': {'Leaf': {'BBA': [0.0, 0.0, 0.15, 0.0, 0.1]}}, 'left': {'Leaf': {'BBA': [0.019230769230769232, 0.09615384615384616, 0.0, 0.07692307692307693, 0.057692307692307696]}}}, 'BBA': [0.010869565217391304, 0.05434782608695652, 0.06521739130434782, 0.043478260869565216, 0.07608695652173914]}}, 'left': {'Leaf': {'BBA': [0.25, 0.0, 0.0, 0.0, 0.0]}}}}}, 'left': {'Leaf': {'BBA': [0.06593406593406594, 0.054945054945054944, 0.04395604395604396, 0.0521978021978022, 0.03296703296703297]}}}, 'BBA': [0.07056451612903226, 0.05040322580645161, 0.04435483870967742, 0.046370967741935484, 0.038306451612903226]}}, 'left': {'IntensityDifference': {319.2286: {'right': {'Leaf': {'BBA': [0.06111111111111111, 0.03611111111111111, 0.041666666666666664, 0.030555555555555555, 0.08055555555555556]}}, 'left': {'gaborSD_3_1': {79.8495: {'right': {'Leaf': {'BBA': [0.0, 0.0, 0.0, 0.0, 0.25]}}, 'left': {'gabormean_2_1': {56.4091: {'right': {'Leaf': {'BBA': [0.03125, 0.140625, 0.078125, 0.0, 0.0]}}, 'left': {'Leaf': {'BBA': [0.1, 0.0, 0.0, 0.15, 0.0]}}}, 'BBA': [0.057692307692307696, 0.08653846153846154, 0.04807692307692308, 0.057692307692307696, 0.0]}}}, 'BBA': [0.041666666666666664, 0.0625, 0.034722222222222224, 0.041666666666666664, 0.06944444444444445]}}}, 'BBA': [0.05555555555555555, 0.04365079365079365, 0.03968253968253968, 0.03373015873015873, 0.07738095238095238]}}}, 'BBA': [0.063, 0.047, 0.042, 0.04, 0.058]}}, 'left': {'Elongation': {1.242119927: {'right': {'SDIntensityBG': {723.0: {'right': {'Entropy': {158000.0: {'right': {'Leaf': {'BBA': [0.25, 0.0, 0.0, 0.0, 0.0]}}, 'left': {'MeanIntensity': {29.0: {'right': {'Leaf': {'BBA': [0.0, 0.11538461538461539, 0.0, 0.038461538461538464, 0.09615384615384616]}}, 'left': {'Leaf': {'BBA': [0.0, 0.025, 0.125, 0.1, 0.0]}}}, 'BBA': [0.0, 0.07608695652173914, 0.05434782608695652, 0.06521739130434782, 0.05434782608695652]}}}, 'BBA': [0.08571428571428572, 0.05, 0.03571428571428571, 0.04285714285714286, 0.03571428571428571]}}, 'left': {'Leaf': {'BBA': [0.06666666666666667, 0.05277777777777778, 0.044444444444444446, 0.05, 0.03611111111111111]}}}, 'BBA': [0.072, 0.052, 0.042, 0.048, 0.036]}}, 'left': {'Sumaverage': {-70300.0: {'right': {'Leaf': {'BBA': [0.07222222222222222, 0.03611111111111111, 0.04722222222222222, 0.03611111111111111, 0.058333333333333334]}}, 'left': {'gaborSD_1_2': {43.1866: {'right': {'RadialDistanceSD': {0.717647059: {'right': {'Leaf': {'BBA': [0.05357142857142857, 0.07142857142857142, 0.017857142857142856, 0.10714285714285714, 0.0]}}, 'left': {'Leaf': {'BBA': [0.15, 0.0, 0.1, 0.0, 0.0]}}}, 'BBA': [0.09375, 0.041666666666666664, 0.052083333333333336, 0.0625, 0.0]}}, 'left': {'Leaf': {'BBA': [0.0, 0.0, 0.0, 0.0, 0.25]}}}, 'BBA': [0.06428571428571428, 0.02857142857142857, 0.03571428571428571, 0.04285714285714286, 0.07857142857142857]}}}, 'BBA': [0.07, 0.034, 0.044, 0.038, 0.064]}}}, 'BBA': [0.071, 0.043, 0.043, 0.043, 0.05]}}}, 'BBA': [0.067, 0.045, 0.0425, 0.0415, 0.054]}}, {'SDIntensityBG': {272.2273: {'right': {'markov2': {231.4831: {'right': {'gaborSD_0_0': {2656.17: {'right': {'Leaf': {'BBA': [0.13202247191011235, 0.056179775280898875, 0.11235955056179775, 0.0702247191011236, 0.12921348314606743]}}, 'left': {'SDIntensity': {435.8908: {'right': {'Leaf': {'BBA': [0.45, 0.0, 0.05, 0.0, 0.0]}}, 'left': {'Area': {141.0: {'right': {'Leaf': {'BBA': [0.0, 0.039473684210526314, 0.15789473684210525, 0.07894736842105263, 0.2236842105263158]}}, 'left': {'Leaf': {'BBA': [0.075, 0.125, 0.225, 0.075, 0.0]}}}, 'BBA': [0.02586206896551724, 0.06896551724137931, 0.1810344827586207, 0.07758620689655173, 0.14655172413793102]}}}, 'BBA': [0.1346153846153846, 0.05128205128205128, 0.14743589743589744, 0.057692307692307696, 0.10897435897435898]}}}, 'BBA': [0.1328125, 0.0546875, 0.123046875, 0.06640625, 0.123046875]}}, 'left': {'markov1': {356.3077: {'right': {'Eccentricity': {1.208467243: {'right': {'Leaf': {'BBA': [0.0, 0.07352941176470588, 0.1323529411764706, 0.11764705882352941, 0.17647058823529413]}}, 'left': {'Leaf': {'BBA': [0.275, 0.075, 0.125, 0.025, 0.0]}}}, 'BBA': [0.10185185185185185, 0.07407407407407407, 0.12962962962962962, 0.08333333333333333, 0.1111111111111111]}}, 'left': {'Leaf': {'BBA': [0.11956521739130435, 0.06521739130434782, 0.10597826086956522, 0.09782608695652174, 0.11141304347826086]}}}, 'BBA': [0.11554621848739496, 0.06722689075630252, 0.11134453781512606, 0.09453781512605042, 0.11134453781512606]}}}, 'BBA': [0.12449392712550607, 0.06072874493927125, 0.11740890688259109, 0.07995951417004049, 0.11740890688259109]}}, 'left': {'IntensityDifference': {204.1443: {'right': {'MaxIntensityBG': {862.0: {'right': {'MeanIntensity': {'BBA': [0.1893939393939394, 0.07575757575757576, 0.08333333333333333, 0.06060606060606061, 0.09090909090909091], 1503.0: {'right': {'Leaf': {'BBA': [0.5, 0.0, 0.0, 0.0, 0.0]}}, 'left': {'gaborSD_3_1': {65.8242: {'right': {'Leaf': {'BBA': [0.125, 0.25, 0.075, 0.025, 0.025]}}, 'left': {'Leaf': {'BBA': [0.0, 0.0, 0.15384615384615385, 0.1346153846153846, 0.21153846153846154]}}}, 'BBA': [0.05434782608695652, 0.10869565217391304, 0.11956521739130435, 0.08695652173913043, 0.13043478260869565]}}}}}, 'left': {'Leaf': {'BBA': [0.16483516483516483, 0.07967032967032966, 0.07417582417582418, 0.06318681318681318, 0.11813186813186813]}}}, 'BBA': [0.17137096774193547, 0.07862903225806452, 0.07661290322580645, 0.0625, 0.11088709677419355]}}, 'left': {'gaborSD_0_0': {19430.96: {'right': {'gabormean_2_1': {47.1436: {'right': {'Perimeter': {164.0: {'right': {'Leaf': {'BBA': [0.0, 0.0, 0.1, 0.325, 0.075]}}, 'left': {'markov1': {379.2439: {'right': {'Leaf': {'BBA': [0.3181818181818182, 0.1590909090909091, 0.022727272727272728, 0.0, 0.0]}}, 'left': {'Leaf': {'BBA': [0.05, 0.125, 0.225, 0.1, 0.0]}}}, 'BBA': [0.19047619047619047, 0.14285714285714285, 0.11904761904761904, 0.047619047619047616, 0.0]}}}, 'BBA': [0.12903225806451613, 0.0967741935483871, 0.11290322580645161, 0.13709677419354838, 0.024193548387096774]}}, 'left': {'Leaf': {'BBA': [0.0, 0.0, 0.05, 0.025, 0.425]}}}, 'BBA': [0.0975609756097561, 0.07317073170731707, 0.0975609756097561, 0.10975609756097561, 0.12195121951219512]}}, 'left': {'Leaf': {'BBA': [0.13068181818181818, 0.07670454545454546, 0.1278409090909091, 0.07386363636363637, 0.09090909090909091]}}}, 'BBA': [0.12015503875968993, 0.0755813953488372, 0.1182170542635659, 0.08527131782945736, 0.10077519379844961]}}}, 'BBA': [0.14525691699604742, 0.07707509881422925, 0.09782608695652174, 0.0741106719367589, 0.10573122529644269]}}}, 'BBA': [0.135, 0.069, 0.1075, 0.077, 0.1115]}}, {'IntensityDifference': {267.0556: {'right': {'markov3': {231.4831: {'right': {'markov4': {371.6891: {'right': {'MinIntensityBG': {481.3424: {'right': {'Leaf': {'BBA': [0.6923076923076923, 0.0, 0.057692307692307696, 0.0, 0.0]}}, 'left': {'gabormean_2_2': {54.3519: {'right': {'Leaf': {'BBA': [0.1, 0.15, 0.475, 0.025, 0.0]}}, 'left': {'gaborSD_3_0': {'BBA': [0.0, 0.03260869565217391, 0.11956521739130435, 0.15217391304347827, 0.44565217391304346], 45.7335: {'right': {'Leaf': {'BBA': [0.0, 0.05, 0.05, 0.3, 0.35]}}, 'left': {'Leaf': {'BBA': [0.0, 0.019230769230769232, 0.17307692307692307, 0.038461538461538464, 0.5192307692307693]}}}}}}, 'BBA': [0.030303030303030304, 0.06818181818181818, 0.22727272727272727, 0.11363636363636363, 0.3106060606060606]}}}, 'BBA': [0.21739130434782608, 0.04891304347826087, 0.1793478260869565, 0.08152173913043478, 0.22282608695652173]}}, 'left': {'SDIntensity': {465.6472: {'right': {'gabormean_1_1': {1604.94: {'right': {'Leaf': {'BBA': [0.75, 0.0, 0.0, 0.0, 0.0]}}, 'left': {'Leaf': {'BBA': [0.6, 0.075, 0.075, 0.0, 0.0]}}}, 'BBA': [0.6818181818181818, 0.03409090909090909, 0.03409090909090909, 0.0, 0.0]}}, 'left': {'MinorAxisLength': {11.86005794: {'right': {'gaborSD_0_1': {5513.7: {'right': {'markov2': {320.1897: {'right': {'gaborSD_1_0': {'BBA': [0.0, 0.0, 0.05952380952380952, 0.13095238095238096, 0.5595238095238095], 367.347: {'right': {'Leaf': {'BBA': [0.0, 0.0, 0.09090909090909091, 0.22727272727272727, 0.4318181818181818]}}, 'left': {'Leaf': {'BBA': [0.0, 0.0, 0.025, 0.025, 0.7]}}}}}, 'left': {'Leaf': {'BBA': [0.0, 0.075, 0.175, 0.225, 0.275]}}}, 'BBA': [0.0, 0.024193548387096774, 0.0967741935483871, 0.16129032258064516, 0.46774193548387094]}}, 'left': {'Leaf': {'BBA': [0.025, 0.125, 0.2, 0.275, 0.125]}}}, 'BBA': [0.006097560975609756, 0.04878048780487805, 0.12195121951219512, 0.18902439024390244, 0.38414634146341464]}}, 'left': {'gabormean_3_1': {64.8078: {'right': {'Leaf': {'BBA': [0.0, 0.3, 0.375, 0.075, 0.0]}}, 'left': {'Leaf': {'BBA': [0.19230769230769232, 0.07692307692307693, 0.40384615384615385, 0.07692307692307693, 0.0]}}}, 'BBA': [0.10869565217391304, 0.17391304347826086, 0.391304347826087, 0.07608695652173914, 0.0]}}}, 'BBA': [0.04296875, 0.09375, 0.21875, 0.1484375, 0.24609375]}}}, 'BBA': [0.2063953488372093, 0.07848837209302326, 0.17151162790697674, 0.11046511627906977, 0.18313953488372092]}}}, 'BBA': [0.21022727272727273, 0.06818181818181818, 0.17424242424242425, 0.10037878787878787, 0.19696969696969696]}}, 'left': {'gabormean_0_2': {1976.31: {'right': {'gaborSD_2_1': {67.25: {'right': {'Leaf': {'BBA': [0.425, 0.2, 0.05, 0.075, 0.0]}}, 'left': {'gaborSD_3_2': {39.6716: {'right': {'Leaf': {'BBA': [0.075, 0.1, 0.35, 0.2, 0.025]}}, 'left': {'Leaf': {'BBA': [0.0, 0.045454545454545456, 0.13636363636363635, 0.11363636363636363, 0.45454545454545453]}}}, 'BBA': [0.03571428571428571, 0.07142857142857142, 0.23809523809523808, 0.15476190476190477, 0.25]}}}, 'BBA': [0.16129032258064516, 0.11290322580645161, 0.1774193548387097, 0.12903225806451613, 0.1693548387096774]}}, 'left': {'Leaf': {'BBA': [0.16758241758241757, 0.08241758241758242, 0.18681318681318682, 0.11263736263736264, 0.20054945054945056]}}}, 'BBA': [0.16598360655737704, 0.09016393442622951, 0.18442622950819673, 0.1168032786885246, 0.19262295081967212]}}}, 'BBA': [0.1889763779527559, 0.07874015748031496, 0.17913385826771652, 0.10826771653543307, 0.19488188976377951]}}, 'left': {'markov1': {203.2727: {'right': {'gabormean_0_2': {'BBA': [0.24152542372881355, 0.1016949152542373, 0.13771186440677965, 0.09322033898305085, 0.17584745762711865], 2008.36: {'right': {'SecondMoment': {1.671967224: {'right': {'Leaf': {'BBA': [0.0, 0.075, 0.025, 0.075, 0.575]}}, 'left': {'Leaf': {'BBA': [0.34375, 0.09375, 0.203125, 0.109375, 0.0]}}}, 'BBA': [0.21153846153846154, 0.08653846153846154, 0.1346153846153846, 0.09615384615384616, 0.22115384615384615]}}, 'left': {'Leaf': {'BBA': [0.25, 0.10597826086956522, 0.13858695652173914, 0.09239130434782608, 0.16304347826086957]}}}}}, 'left': {'gabormean_0_1': {19588.11: {'right': {'gabormean_2_0': {33.0867: {'right': {'MaxIntensityBG': {'BBA': [0.19827586206896552, 0.10344827586206896, 0.25862068965517243, 0.15517241379310345, 0.034482758620689655], 412.9867: {'right': {'Leaf': {'BBA': [0.475, 0.125, 0.15, 0.0, 0.0]}}, 'left': {'Leaf': {'BBA': [0.05263157894736842, 0.09210526315789473, 0.3157894736842105, 0.23684210526315788, 0.05263157894736842]}}}}}, 'left': {'Leaf': {'BBA': [0.0, 0.0, 0.075, 0.075, 0.6]}}}, 'BBA': [0.14743589743589744, 0.07692307692307693, 0.21153846153846154, 0.1346153846153846, 0.1794871794871795]}}, 'left': {'Leaf': {'BBA': [0.19662921348314608, 0.10112359550561797, 0.17415730337078653, 0.10393258426966293, 0.17415730337078653]}}}, 'BBA': [0.181640625, 0.09375, 0.185546875, 0.11328125, 0.17578125]}}}, 'BBA': [0.21036585365853658, 0.0975609756097561, 0.16260162601626016, 0.10365853658536585, 0.1758130081300813]}}}, 'BBA': [0.1995, 0.088, 0.171, 0.106, 0.1855]}}, {'markov1': {267.0556: {'right': {'gabormean_0_2': {1631.7: {'right': {'gabormean_1_0': {2199.9: {'right': {'MaxIntensity': {1412.0: {'right': {'Leaf': {'BBA': [0.875, 0.025, 0.025, 0.025, 0.05]}}, 'left': {'gabormean_2_0': {'BBA': [0.05172413793103448, 0.19827586206896552, 0.22413793103448276, 0.22413793103448276, 0.3017241379310345], 57.0641: {'right': {'Leaf': {'BBA': [0.1, 0.325, 0.275, 0.275, 0.025]}}, 'left': {'Leaf': {'BBA': [0.02631578947368421, 0.13157894736842105, 0.19736842105263158, 0.19736842105263158, 0.4473684210526316]}}}}}}, 'BBA': [0.26282051282051283, 0.15384615384615385, 0.17307692307692307, 0.17307692307692307, 0.23717948717948717]}}, 'left': {'SDIntensity': {444.6014: {'right': {'Solidity': {0.977777778: {'right': {'Leaf': {'BBA': [1.0, 0.0, 0.0, 0.0, 0.0]}}, 'left': {'Leaf': {'BBA': [0.9, 0.0, 0.075, 0.025, 0.0]}}}, 'BBA': [0.9545454545454546, 0.0, 0.03409090909090909, 0.011363636363636364, 0.0]}}, 'left': {'Contrast': {43.5545: {'right': {'MinorAxisLength': {9.194970613999999: {'right': {'markov4': {74.0: {'right': {'Leaf': {'BBA': [0.09375, 0.296875, 0.390625, 0.140625, 0.078125]}}, 'left': {'Leaf': {'BBA': [0.05, 0.1, 0.35, 0.45, 0.05]}}}, 'BBA': [0.07692307692307693, 0.22115384615384615, 0.375, 0.25961538461538464, 0.0673076923076923]}}, 'left': {'Leaf': {'BBA': [0.1, 0.525, 0.375, 0.0, 0.0]}}}, 'BBA': [0.08333333333333333, 0.3055555555555556, 0.375, 0.1875, 0.04861111111111111]}}, 'left': {'gabormean_1_2': {34.2439: {'right': {'MinorAxisLength': {39.54886118: {'right': {'Leaf': {'BBA': [0.0, 0.025, 0.0, 0.2, 0.775]}}, 'left': {'Leaf': {'BBA': [0.0, 0.022727272727272728, 0.22727272727272727, 0.38636363636363635, 0.36363636363636365]}}}, 'BBA': [0.0, 0.023809523809523808, 0.11904761904761904, 0.2976190476190476, 0.5595238095238095]}}, 'left': {'Leaf': {'BBA': [0.0, 0.025, 0.225, 0.0, 0.75]}}}, 'BBA': [0.0, 0.024193548387096774, 0.1532258064516129, 0.20161290322580644, 0.6209677419354839]}}}, 'BBA': [0.04477611940298507, 0.17537313432835822, 0.27238805970149255, 0.19402985074626866, 0.31343283582089554]}}}, 'BBA': [0.2696629213483146, 0.13202247191011235, 0.21348314606741572, 0.14887640449438203, 0.23595505617977527]}}}, 'BBA': [0.267578125, 0.138671875, 0.201171875, 0.15625, 0.236328125]}}, 'left': {'gaborSD_0_1': {756438.56: {'right': {'MeanIntensity': {1578.0: {'right': {'Leaf': {'BBA': [0.7692307692307693, 0.057692307692307696, 0.17307692307692307, 0.0, 0.0]}}, 'left': {'gabormean_2_2': {49.5579: {'right': {'Leaf': {'BBA': [0.0, 0.2, 0.425, 0.2, 0.175]}}, 'left': {'Leaf': {'BBA': [0.0, 0.057692307692307696, 0.17307692307692307, 0.19230769230769232, 0.5769230769230769]}}}, 'BBA': [0.0, 0.11956521739130435, 0.2826086956521739, 0.1956521739130435, 0.40217391304347827]}}}, 'BBA': [0.2777777777777778, 0.09722222222222222, 0.24305555555555555, 0.125, 0.2569444444444444]}}, 'left': {'Leaf': {'BBA': [0.23055555555555557, 0.14722222222222223, 0.26666666666666666, 0.14166666666666666, 0.21388888888888888]}}}, 'BBA': [0.24404761904761904, 0.13293650793650794, 0.25992063492063494, 0.13690476190476192, 0.2261904761904762]}}}, 'BBA': [0.2559055118110236, 0.13582677165354332, 0.23031496062992127, 0.1466535433070866, 0.2312992125984252]}}, 'left': {'markov2': {202.6316: {'right': {'MaxIntensityBG': {'BBA': [0.30952380952380953, 0.15476190476190477, 0.20436507936507936, 0.125, 0.20634920634920634], 847.0: {'right': {'MeanIntensity': {'BBA': [0.2847222222222222, 0.14583333333333334, 0.2013888888888889, 0.11805555555555555, 0.25], 1503.0: {'right': {'Leaf': {'BBA': [0.825, 0.0, 0.05, 0.05, 0.075]}}, 'left': {'Area': {'BBA': [0.07692307692307693, 0.20192307692307693, 0.25961538461538464, 0.14423076923076922, 0.3173076923076923], 151.0: {'right': {'Leaf': {'BBA': [0.0, 0.078125, 0.1875, 0.234375, 0.5]}}, 'left': {'Leaf': {'BBA': [0.2, 0.4, 0.375, 0.0, 0.025]}}}}}}}}, 'left': {'MaxIntensity': {1437.0: {'right': {'Elongation': {1.1764839790000001: {'right': {'Leaf': {'BBA': [1.0, 0.0, 0.0, 0.0, 0.0]}}, 'left': {'Leaf': {'BBA': [0.95, 0.0, 0.05, 0.0, 0.0]}}}, 'BBA': [0.98, 0.0, 0.02, 0.0, 0.0]}}, 'left': {'gabormean_3_1': {42.102: {'right': {'gaborSD_3_1': {62.2908: {'right': {'Leaf': {'BBA': [0.325, 0.35, 0.325, 0.0, 0.0]}}, 'left': {'ConvexPerimeter': {49.20108142: {'right': {'Leaf': {'BBA': [0.0, 0.2, 0.225, 0.45, 0.125]}}, 'left': {'gaborSD_2_2': {54.8611: {'right': {'Leaf': {'BBA': [0.038461538461538464, 0.5192307692307693, 0.28846153846153844, 0.15384615384615385, 0.0]}}, 'left': {'Leaf': {'BBA': [0.05, 0.2, 0.575, 0.15, 0.025]}}}, 'BBA': [0.043478260869565216, 0.3804347826086957, 0.41304347826086957, 0.15217391304347827, 0.010869565217391304]}}}, 'BBA': [0.030303030303030304, 0.32575757575757575, 0.3560606060606061, 0.24242424242424243, 0.045454545454545456]}}}, 'BBA': [0.09883720930232558, 0.3313953488372093, 0.3488372093023256, 0.18604651162790697, 0.03488372093023256]}}, 'left': {'RadialDistanceSD': {3.22410917: {'right': {'Leaf': {'BBA': [0.0, 0.0, 0.0, 0.35, 0.65]}}, 'left': {'Leaf': {'BBA': [0.0, 0.0, 0.25, 0.0, 0.75]}}}, 'BBA': [0.0, 0.0, 0.13636363636363635, 0.1590909090909091, 0.7045454545454546]}}}, 'BBA': [0.06538461538461539, 0.21923076923076923, 0.27692307692307694, 0.17692307692307693, 0.26153846153846155]}}}, 'BBA': [0.3194444444444444, 0.15833333333333333, 0.20555555555555555, 0.12777777777777777, 0.18888888888888888]}}}}}, 'left': {'gabormean_1_2': {67.4787: {'right': {'Leaf': {'BBA': [0.25824175824175827, 0.1620879120879121, 0.21978021978021978, 0.15934065934065933, 0.20054945054945056]}}, 'left': {'Clustertendency': {0.027999999999999997: {'right': {'Leaf': {'BBA': [0.013888888888888888, 0.20833333333333334, 0.2916666666666667, 0.2638888888888889, 0.2222222222222222]}}, 'left': {'Leaf': {'BBA': [0.6363636363636364, 0.11363636363636363, 0.25, 0.0, 0.0]}}}, 'BBA': [0.25, 0.1724137931034483, 0.27586206896551724, 0.16379310344827586, 0.13793103448275862]}}}, 'BBA': [0.25625, 0.16458333333333333, 0.23333333333333334, 0.16041666666666668, 0.18541666666666667]}}}, 'BBA': [0.28353658536585363, 0.15955284552845528, 0.2184959349593496, 0.14227642276422764, 0.19613821138211382]}}}, 'BBA': [0.2695, 0.1475, 0.2245, 0.1445, 0.214]}}]
    ## Get actual data
    actualTrain = getPigns(train_features,4)
    actualTest = getPigns(test_features,4)
    actualCalib = getPigns(calib_features,4)
    
    # Classify calibration set & compute conformity
    print ("Classifying Calibration Set...") 
    calib_label_dct, calibLabels = classifyIterativeTrees(trees,test_header,calib_features)
    print(len(calibLabels))
    # Calibration Conformity
    print("Computing Calibration Conformity...")
    print(calibLabels)
    print(actualCalib)
    calib_conf = Train_Conformity(actualCalib, calibLabels)
    
    print("\nCALIBRATION:", file=f)
    for i in range (0,len(actualCalib)):
        print("Probabilities: ", actualCalib[i], "\tConformity Score = ", calib_conf[i], file=f)
        
    train_label_dict, trnLabels = classifyIterativeTrees(trees,test_header,train_features)
    test_label_dict, tstLabels = classifyIterativeTrees(trees, test_header, test_features)
    modified_avg_train = getAverageBBA(train_label_dict)
    modified_max_train = getMaxBBA(train_label_dict)
    modified_avg_test = getAverageBBA(test_label_dict)
    modified_max_test = getMaxBBA(test_label_dict)
    modified_con_test = getConfidentBBA(test_label_dict,calib_conf)
    
    selective_max = getMaxy([modified_avg_test, modified_max_test, modified_con_test])
#    selective_modal = getMode([modified_avg_test, modified_max_test, modified_con_test])
    selective_avg = getAvy([modified_avg_test, modified_max_test, modified_con_test])
    actual_one = getPigns(test_features,1)
    actual_two = getPigns(test_features,2)
    actual_three = getPigns(test_features,3)
    one = test_label_dict[1]
    two = test_label_dict[2]
    three = test_label_dict[3]
    four = test_label_dict[4]
    
    tst_proba = [getMax(case)+1 for case in actualTest]
    
    modal_avg_matrix = getConfusionMatrix(selective_max, actualTest)
    test_accy = getAccuracy(modal_avg_matrix, len(actualTest))
    
    print("\nTESTING CONFUSION MATRIX SELECTIVE MAX BBA", file=f)
    print(modal_avg_matrix,file=f)
    print("\nTESTING ACCURACY = ", test_accy, file=f)  

    modal_avg_matrix = getConfusionMatrix(selective_avg, actualTest)
    test_accy = getAccuracy(modal_avg_matrix, len(actualTest))
    
    print("\nTESTING CONFUSION MATRIX SELECTIVE AVG BBA", file=f)
    print(modal_avg_matrix,file=f)
    print("\nTESTING ACCURACY = ", test_accy, file=f)
    
#    modal_matrix = confusion_matrix(selective_modal,tst_proba)
#    test_accy = getAccuracy(modal_matrix, len(actualTest))
#    
#    print("\nTESTING CONFUSION MATRIX SELECTIVE MODAL BBA", file=f)
#    print(modal_matrix,file=f)
#    print("\nTESTING ACCURACY = ", test_accy, file=f)
    #print (trees)
    #print("BDT TREE: ",file =f)
    #print(tree, file = f)
    ##createPlot(tree)
    #
    
        
    #for i in range(1,5):
    #    out.append(calcEntrop(train_features,i))
    #    out2.append(calcEntrop(test_features,i))
    #actualTrain = []
    #actualTest = []
    #for i in range(4):
    #    actualTrain.append(out[i][2])
    #    actualTest.append(out2[i][2])
    #    
    ## Classify training set
    #for i in range(0,len(train_features)):
    #    trainLabels.append(classify(tree, test_header,train_features[i]))
     #Get training accuracy
        
    #training_matricies = []
    
    #print(modified_train)   
    #print(len(modified_train))     
        #training_matricies.append(getConfusionMatrix(trainLabels,))
    #    
    #print (training_matricies)
    #train_class_matrix = getConfusionMatrix(modified_avg_train, actualTrain)
    #train_accy = getAccuracy(train_class_matrix, len(trainLabels))
    #print("\nTRAINING CONFUSION MATRIX",file=f)
    #print(train_class_matrix,file=f)
    #print("\n AVG BBA TRAINING ACCURACY = ", train_accy, file=f)
    #
    #train_class_matrix = getConfusionMatrix(modified_max_train, actualTrain)
    #train_accy = getAccuracy(train_class_matrix, len(trainLabels))
    #print("\nTRAINING CONFUSION MATRIX",file=f)
    #print(train_class_matrix,file=f)
    #print("\n MAX BBA TRAINING ACCURACY = ", train_accy, file=f)
    
    ## Classify testing set
    #for i in range(0,len(test_features)):
    #    testLabels.append(classify(tree, test_header,test_features[i]))
    test_class_matrix = getConfusionMatrix(modified_avg_test, actualTest)
    test_accy = getAccuracy(test_class_matrix, len(tstLabels))
    
    # Get testing accuracy
    print("\nTESTING CONFUSION MATRIX AVG BBA", file=f)
    print(test_class_matrix,file=f)
    print("\nTESTING ACCURACY = ", test_accy, file=f)
    
    print(len(modified_max_test),len(actualTest))
    test_class_matrix = getConfusionMatrix(modified_max_test, actualTest)
    test_accy = getAccuracy(test_class_matrix, len(tstLabels))
    
    # Get testing accuracy
    print("\nTESTING CONFUSION MATRIX MAX BBA", file=f)
    print(test_class_matrix,file=f)
    print("\nTESTING ACCURACY = ", test_accy, file=f)
    
    test_class_matrix = getConfusionMatrix(modified_con_test, actualTest)
    test_accy = getAccuracy(test_class_matrix, len(tstLabels))
    
    # Get testing accuracy
    print("\nTESTING CONFUSION MATRIX BEST CONFIDENCE", file=f)
    print(test_class_matrix,file=f)
    print("\nTESTING ACCURACY = ", test_accy, file=f)
    
    test_class_matrix = getConfusionMatrix(one, actual_one)
    test_accy = getAccuracy(test_class_matrix, len(tstLabels))
    
    # Get testing accuracy
    print("\nTESTING CONFUSION MATRIX LABEL ONE", file=f)
    print(test_class_matrix,file=f)
    print("\nTESTING ACCURACY = ", test_accy, file=f)
    
    test_class_matrix = getConfusionMatrix(two, actual_two)
    test_accy = getAccuracy(test_class_matrix, len(tstLabels))
    
    # Get testing accuracy
    print("\nTESTING CONFUSION MATRIX LABEL TWO", file=f)
    print(test_class_matrix,file=f)
    print("\nTESTING ACCURACY = ", test_accy, file=f)
    
    test_class_matrix = getConfusionMatrix(three, actual_three)
    test_accy = getAccuracy(test_class_matrix, len(tstLabels))
    
    # Get testing accuracy
    print("\nTESTING CONFUSION MATRIX LABEL THREE", file=f)
    print(test_class_matrix,file=f)
    print("\nTESTING ACCURACY = ", test_accy, file=f)
    
    test_class_matrix = getConfusionMatrix(four, actualTest)
    test_accy = getAccuracy(test_class_matrix, len(tstLabels))
    
    # Get testing accuracy
    print("\nTESTING CONFUSION MATRIX LABEL FOUR", file=f)
    print(test_class_matrix,file=f)
    print("\nTESTING ACCURACY = ", test_accy, file=f)
    
    
    test_conf = Test_Conformity(selective_avg)
    p_vals = PValue(test_conf,calib_conf)
    confidence = []
    credibility = []
    for i, val in enumerate(p_vals):
        m = val.index(max(val))
        sec = max([x for k,x in enumerate(val) if k != m])
    #    print(i, "  P-values: ", val, "  Class Label: ", val.index(max(val))+1, "  Confidence: ", 1-sec, "\tCredibility: ", max(val))
        confidence.append(1 - sec)
        credibility.append(max(val))
        
    print ("Writing Data...") 
    trainhead = "Training Data: (d = " + str(maxdepth) + " | np = " + str(nparent) + " | nc = " + str(nchild) + ")"
    testhead = "Testing Data: (d = " + str(maxdepth) + " | np = " + str(nparent) + " | nc = " + str(nchild) + ")"
    writeData("Training", trainhead, "../output/TrainOutput.csv", "wb", actualTrain, trnLabels, confidence, credibility, 0)
    writeData("Testing", testhead, "../output/TestOutput1.csv", "wb", actualTest, selective_avg , confidence, credibility, len(trnLabels))
f.close()
#
## Output training data to csv
#with open("output/TrainOutput.csv", "wb") as csvfile:
#    writer = csv.writer(csvfile, delimiter=',',
#                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
#    writer.writerow(['Actual Training [1]', 'Actual Training [2]', 'Actual Training [3]', 'Actual Training [4]', 'Actual Training [5]', 'Training [1]' , 'Training [2]', 'Training [3]', 'Training [4]', 'Training [5]', 'RMSE', 'Pearson r'])
#    for i in range(0, len(trainLabels)):
#        # Calculate RMSE
#        rmse = 0
#        for j in range (0,5):
#            rmse += pow((actualTrain[i][j] - trainLabels[i][j]),2)
#        rmse /= 5
#        rmse = math.sqrt(rmse)
#        
#        # Calculate Pearson correlation coefficient (r)
#        r = 0
#        xySum = 0
#        xSum = 0
#        ySum = 0
#        for j in range (0,5):        
#            xySum += (actualTrain[i][j] - sum(actualTrain[i])/5) * (trainLabels[i][j] - sum(trainLabels[i])/5)
#            xSum += pow((actualTrain[i][j] - sum(actualTrain[i])/5),2)
#            ySum += pow((trainLabels[i][j] - sum(trainLabels[i])/5),2)
#        r = xySum / (math.sqrt(xSum) * math.sqrt(ySum))
#        
#        writer.writerow([actualTrain[i][0], actualTrain[i][1], actualTrain[i][2], actualTrain[i][3], actualTrain[i][4], trainLabels[i][0], trainLabels[i][1], trainLabels[i][2], trainLabels[i][3], trainLabels[i][4], rmse, r])
#
## Output training data to csv
#with open("output/TestOutput.csv", "wb") as csvfile:
#    writer = csv.writer(csvfile, delimiter=',',
#                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
#    writer.writerow(['Actual Testing [1]', 'Actual Testing [2]', 'Actual Testing [3]', 'Actual Testing [4]', 'Actual Testing [5]', 'Testing [1]', 'Testing [2]', 'Testing [3]', 'Testing [4]', 'Testing [5]', 'RMSE', 'Pearson r'])    
#    for i in range(0, len(testLabels)):
#        # Calculate RMSE        
#        rmse = 0
#        for j in range (0,5):
#            rmse += pow((actualTest[i][j] - testLabels[i][j]),2)
#        rmse /= 5
#        rmse = math.sqrt(rmse)
#
#        # Calculate Pearson correlation coefficient (r)
#        r = 0
#        xySum = 0
#        xSum = 0
#        ySum = 0
#        for j in range (0,5):        
#            xySum += (actualTest[i][j] - sum(actualTest[i])/5) * (testLabels[i][j] - sum(testLabels[i])/5)
#            xSum += pow((actualTest[i][j] - sum(actualTest[i])/5),2)
#            ySum += pow((testLabels[i][j] - sum(testLabels[i])/5),2)
#        r = xySum / (math.sqrt(xSum) * math.sqrt(ySum))        
#        
#        writer.writerow([actualTest[i][0], actualTest[i][1], actualTest[i][2], actualTest[i][3], actualTest[i][4], testLabels[i][0], testLabels[i][1], testLabels[i][2], testLabels[i][3], testLabels[i][4], rmse, r])
