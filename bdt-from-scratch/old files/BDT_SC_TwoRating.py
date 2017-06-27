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
    LIDC_Data  = pd.read_csv(set_name, header =0)
    header = list(LIDC_Data.columns.values)[:-4]
    LIDC_Data = LIDC_Data._get_numeric_data()
    LIDC_Data = LIDC_Data.as_matrix()
    return LIDC_Data, header  
#def importData(test_name, train_name, calib_name):
#    global header  
#    global tst_data_array
#    global trn_data_array
##    global cal_data_array
#    
#    tst_f =  test_name # import file
#    tst_df = pd.read_csv(tst_f, header = 0)
#    header = list(tst_df.columns.values)[:-4]
#    tst_df = tst_df._get_numeric_data()
#    tst_data_array = tst_df.as_matrix()
#    
#    trn_f =  train_name # import file
#    trn_df = pd.read_csv(trn_f, header = 0)
#    trn_df = trn_df._get_numeric_data()
#    trn_data_array = trn_df.as_matrix()
    
#    cal_f =  calib_name # import file
#    cal_df = pd.read_csv(cal_f, header = 0)
#    header = list(cal_df.columns.values)[:-4]
#    cal_df = cal_df._get_numeric_data()
#    cal_data_array = cal_df.as_matrix()
    
# SET TRAINING DATA
def setTrain(lst):

    #rand_trn = [ trn_data_array[i].tolist() for i in random.sample(xrange(len(trn_data_array)), len(trn_data_array)) ]
    #np.random.shuffle(trn_data_array)
    split = int((6.0/7.0) * len(lst))
    #print(split)
    train_features = lst[:split].tolist() # training data
    #calib_features = [ train_features[i] for i in sorted(random.sample(xrange(len(train_features)), split)) ]
    calib_features = lst[split:].tolist() # calibration data
    return train_features, calib_features
# EXTRACT DATA
def getPigns(dataset, ratings):
    nodePigns = []
    # For each case at this node, calculate a vector of probabilities
    for case in dataset:
        if ratings == 4:
            currentLabel = case[-4:]
        else:
            currentLabel = case[-4:-4+ratings]
        zeroratings = 0
        pign = []
        
        # Convert radiologist ratings into pignistic probability distributions
        for i in range (0, 6): # Count number of instances of each rating
            if i == 0:
                zeroratings += currentLabel.count(i)
            else:
                pign.append(currentLabel.count(i))
        
        pign[:] = [float(x) / (ratings - zeroratings) for x in pign] 
        pign = five_to_two(pign)
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
    Apign = [0.0, 0.0]
    for k in range (0, len(nodePigns)): # Sum of probabilities for each rating
        for i in range (0, 2): 
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
    for k in range (0, 2):
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
                if splitInfo == 0: pass
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
#            print ("Stopping conditions met",file=f)
#            print ("LEAF >>>>>>>>>>>\n",file=f)
            decision_tree = {"Leaf":{"BBA":baseApign}}
            return decision_tree
        elif (bestGainRatio == 0):
#            print ("Found leaf node with Gain = 0",file=f)
#            print ("LEAF >>>>>>>>>>>\n",file=f)
            decision_tree = {"Leaf":{"BBA":baseApign}}
            return decision_tree
        elif (all_same(basePigns)):
#            print ("Found leaf node with all BBA's equal",file=f)
#            print ("LEAF >>>>>>>>>>>\n",file=f)
            decision_tree = {"Leaf":{"BBA":baseApign}}
            return decision_tree
        elif (curr_depth == max_depth): 
#            print ("Max depth reached: ",str(max_depth),file=f)
#            print ("LEAF >>>>>>>>>>>\n",file=f)
            decision_tree = {"Leaf":{"BBA":baseApign}}
            return decision_tree
            
        # Recursive Call
        else:
#            print ("\tSplitting at depth ", curr_depth, "...")  
#            print ("Length of label set = " + str(len(labels)),file=f)
#            print ("Selected feature to split = " + str(bestFeatLabel) + ", " + str(bestFeat),file=f)
#            print ("Best value = " + str(bestVal),file=f)
#            print ("bestGainRatio = " + str(bestGainRatio), file=f)            
            
            # Create a split
            decision_tree[bestFeatLabel][bestVal] = {}            
            subLabels = labels[:]
            lowSet, highSet = splitData(dataset, bestFeat, bestVal) # returns low & high set
#            print ("Set Lengths (low/high) = ",str(len(lowSet))," / ",str(len(highSet)),file=f)
#            print("SPLIT >>>>>>>>>>>\n", file=f)            
            
            # Recursively create child nodes
            decision_tree[bestFeatLabel][bestVal]["left"] = createTree(lowSet, subLabels, min_parent, min_child,\
                                                                       curr_depth+1, max_depth,num_ratings)
            decision_tree[bestFeatLabel][bestVal]["right"] = createTree(highSet, subLabels, min_parent, min_child,\
                                                                        curr_depth+1, max_depth,num_ratings)
            
            return decision_tree
    
    # If no labels left, then STOP
    elif (len(labels) < 1):
#        print ("All features have been used to split; all further nodes are leaves", file=f)
#        print("LEAF >>>>>>>>>>>\n",file=f)
        return
#####################################
# OUTPUT RESULTS DATAFILES
#####################################
def writeData(trainOrTest, header, filename, params, actual, predicted, conf, cred, id_start):
    avgROC = 0    
    ratings = 2.0
    with open(filename, params) as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if trainOrTest == "Training":
            writer.writerow([
                             'Actual [1]',      'Actual [2]',\
                             'Predicted [1]' ,  'Predicted [2]',\
                             'RMSE', 'Pearson r', 'ROC AUC', header])
        else:
            writer.writerow([
                             'Actual [1]',      'Actual [2]', \
                             'Predicted [1]' ,  'Predicted [2]',\
                             'RMSE', 'Pearson r', 'ROC AUC', 'Confidence', 'Credibility', header])
        
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
            for j in range(2):
                rmse += pow((actual[i][j] - predicted[i][j]), 2)
                xySum += (actual[i][j] - sum(actual[i])/ratings) * (predicted[i][j] - sum(predicted[i])/5)
                xSum += pow((actual[i][j] - sum(actual[i])/ratings),2)
                ySum += pow((predicted[i][j] - sum(predicted[i])/ratings),2)
            rmse /= ratings
            rmse = math.sqrt(rmse)
#            r = xySum / (math.sqrt(xSum) * math.sqrt(ySum))
            
            r = 0.0
#            rmse = 0.0
#            roc = 0.0
            # Write data and similarity measures to file
            if trainOrTest == "Training":
                writer.writerow([
                                 actual[i][0], actual[i][1],\
                                 predicted[i][0], predicted[i][1],\
                                 rmse, r, roc])
            else:
                writer.writerow([
                                 actual[i][0], actual[i][1],\
                                 predicted[i][0], predicted[i][1], \
                                 rmse, r, roc, conf[i], cred[i]])
            
    # Calculate Accuracy and AUCdt
    conf_matrix = getConfusionMatrix(predicted, actual)
    accy = getAccuracy(conf_matrix)
    cMatrix = modConfusion(predicted, actual)
    modAccy = getAccuracy(cMatrix)
    myAUCdt = scipy.integrate.quad(AUCdt, 0, 1, args = (actual, predicted))
    avgROC /= len(predicted)
    cos_sim, similarities, ratings = CosineSimilarity(actual, predicted)
    
    # Output Confusion Matrices, Accuracies, AUCdt, and ROC AUC
    print("\n", trainOrTest, "Confusion Matrix", file=f)
    print(conf_matrix,file=f)
    print("\n", trainOrTest, "MOD Confusion Matrix", file=f)
    np.set_printoptions(precision=3)
    print("",np.asarray(cMatrix[0]),"\n",np.asarray(cMatrix[1]),"\n", "\n",file=f)
    print(" Matrix Sum = ",sum2D(cMatrix),"\n",file=f)
    print("Accuracy = ", '{:.4}'.format(accy * 100), "%", file=f)
    print("MOD Accuracy = ", '{:.4}'.format(modAccy * 100), "%", file=f)
    print("AUCdt JD = ", '{:.4}'.format(myAUCdt[0]), " with error of ", '{:.4}'.format(myAUCdt[1]), file=f)
    print("Avg ROC AUC = ", '{:.4}'.format(avgROC), file=f)
    
#####################################
# CONFORMAL PREDICTION
#####################################
# Sum elements in a 2D array
def sum2D(input):
    return sum(map(sum, input))

# MODIFIED ACCURACY EQUATION
def modConfusion(pred, actual):
    matrix = [[0,0],
              [0,0]]
    
    for i in range (0,len(pred)):
        dot = [[0,0],
              [0,0]]
        # Multiply elements of actual and predicted together to form a 5x5 matrix
        for j in range (0,2):
            for k in range (0,2):
                dot[j][k] = (float(actual[i][j]) * float(pred[i][k]))
        
        # Add said matrix to final matrix
        for j in range (0,2):
            for k in range (0,2):
                temp = matrix[j][k]
                matrix[j][k] = temp + dot[j][k]
    # Normalize matrix
    for j in range (0,2):
        for k in range (0,2):
            matrix[j][k] /= len(pred)           
    #print (matrix)
    return matrix
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
    
def Train_Conformity(actual, predicted):
    conform = []
    for i in range(len(actual)):
        max_ind = getMax(actual[i]) #index of max actual
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
        counts = [0,0] #stores count of test conformity greater than equal to calib 
        for j in range(0,2): #for each label in test
            val = test[i][j] #conformity value of label
            for k in range(0,len(calib)): #for each conformity value 
                if calib[k] <= val: #if less than val increase count of label
                    counts[j] += 1
        pvalues.append([count/float(len(calib)+1) for count in counts]) #append p values for case
    return pvalues
    
#####################################
# CONFUSION MATRIX BUILDING
#####################################
def getConfusionMatrix(predicted,actual): #returns misclassification matrix
    # TEST: Find greatest probability to use as prediction
    pred_proba = [getMax(case)+1 for case in predicted]
    act_proba = [getMax(case)+1 for case in actual]
    return confusion_matrix(act_proba,pred_proba)
    
def getAccuracy(class_matrix): # returns accuracy of misclassification matrix 
    accy = 0.0
    for j in range(2):
        accy += class_matrix[j][j]
            
    return accy / sum2D(class_matrix)  
    
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
    binaryActual = [0, 0]
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
        
#############################
#SELECTIVE EVALUATION METHODS
#############################

#RETURNS INDEX OF MODEL THAT HAS BBA VECTOR WITH CONFIDENCE VALUE ABOVE THRESHOLD
def highestConfidence(lst):
    for i, val in enumerate(lst):
        m = getMax(val)
        conf = 1 - max([x for k,x in enumerate(val) if k != m])
        if conf >= .875 and i != 0:
            return i
        elif conf >= .95:
            return i
            
    return -1
      
# function adds weight to a bba **unused**
def addWeight(lst,weight):
    for i in range(len(lst)):
        lst[i] = lst[i] * weight
    return lst
    
#selective iterative classification method. 
#selects model3 based on confidence, max bba value in a vector, and averages the bbas if no model is satisfactory.
def selectiveChoice(label_dict,calib):
    modified_labels = []
    used_trees = [0,0,0,0,0]
    for j in range(0,len(label_dict[1])): #computes the average bba values across all four models 
        bbas = []
        avg_bba = [0.0,0.0]
        for i in range(1,5): #4 model bbas for one case
                bbas.append(label_dict[i][j])
                for k in range(2):
                    avg_bba[k] += label_dict[i][j][k] #sum of probabilities
        for i in range(len(avg_bba)): #divided by number of trees
            avg_bba[i] = avg_bba[i]/4.0
         
        #calculate bba confidence
        conform = Test_Conformity(bbas)
        p_values = PValue(conform, calib)  
        temp = highestConfidence(p_values) #if higher than .9 return index of bba vector
        if temp != -1:           
            used_trees[temp] += 1
            choice = bbas[temp]
#            modified_labels.append(choice)
        else: #compute max probability
            choice = []
            for bba in bbas:
                if max(bba) >= .75:
                    ind = bba.index(max(bba))
                    used_trees[ind] += 1
                    choice = bba
                    break
            if len(choice) == 0: #compute average across all models 
                used_trees[-1] += 1
                choice = avg_bba
        modified_labels.append(choice)
    print ("Selective choice methods distribution: ", used_trees)
    return modified_labels
        
#selective iterative method that chooses  model based on confidence of the bba            
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
        if temp != -1:           
            used_trees[temp] += 1
            choice = bbas[temp]
            modified_train.append(choice)
        else:
            avg_bba = [0.0,0.0]
            for val in bbas:         
                ####CODE FOR AVERAGE BBA VALUES 
                for k in range(2):
                    avg_bba[k] += val[k]
            for i in range(len(avg_bba)):
                avg_bba[i] = avg_bba[i]/4.0
            modified_train.append(avg_bba)
    print("\nDistribution of trees used: ", used_trees)
    return modified_train    
 
#averages the bbas across all models   
def getAverageBBA(trn_label_dict):
    modified_train = []
    for j in range(0,len(trn_label_dict[1])): #computes the average bba values across all four models 
        avg_bba = [0.0,0.0]
        for i in range(1,5):            
            ####CODE FOR AVERAGE BBA VALUES 
            for k in range(2):
                avg_bba[k] += trn_label_dict[i][j][k]
        for i in range(len(avg_bba)):
            avg_bba[i] = avg_bba[i]/4.0
        modified_train.append(avg_bba)
    return modified_train
    
def getMaxBBA(trn_label_dict):
    modified_train = []
    dist = [0,0,0,0,0]
    l = 0
    for j in range(0,len(trn_label_dict[1])): #computes the average bba values across all four models 
        max_prob = 0
        avg_bba = [0.0,0.0]
        for i in range(1,5):
            if max(trn_label_dict[i][j]) >= .75 and i != 1:
                avg_bba = trn_label_dict[i][j]
                l = i
                break
            elif max(trn_label_dict[i][j]) >= .85:
                avg_bba = trn_label_dict[i][j]
                l = i
                break
            elif(max(trn_label_dict[i][j]) > max_prob):
                max_prob = max(trn_label_dict[i][j])
                avg_bba = trn_label_dict[i][j]
                l = i
        dist[l] += 1
        modified_train.append(avg_bba)
    print ("max: ",dist)
    return modified_train        
       
#computes max bba across three lists of bbas       
def getMaxy(lst):
    result = []
    for i in range(len(lst[0])):
        maxy = [0.0,0.0]
        max_val = 0
        for k in range(len(lst)):
            temp = max(lst[k][i])
            if temp > max_val:
                max_val = temp
                maxy = lst[k][i]
        result.append(maxy)
    return result

#computes averages from lists of list of bbas 
def getAvy(lst):
    result = []
    for i in range(len(lst[0])):
        subavg = [0.0,0.0]
        for k in range(len(lst)):
            for j in range(len(lst[0][0])):
                subavg[j] += lst[k][i][j]
        subavg = [y/3 for y in subavg]
        result.append(subavg)
    return result
    
#selects mode classifcation label  from lst of bbas
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
def classifyIterativeTrees(trees,data_header,train_features):
    train_label_dct = {}
    Labels = []

    for k in range(0,4): #builds dictionary of bba labels for all four models 
        Labels = []
        for i in range(0,len(train_features)):
            Labels.append(classify(trees[k], data_header,train_features[i]))
#        if k in train_label_dct:
#            train_label_dct[k+1].update(Labels)
#        else:
        train_label_dct[k+1] = Labels
            
    return train_label_dct, Labels
    
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
            
def buildTrees(head, train_features, parent, child, depth):
    trees = []
    parents = [16,16,16,16]
    children = [8,8,8,8]
    tf = train_features
    hdr = ['Area', 'ConvexArea', 'Perimeter', 'ConvexPerimeter', 'EquivDiameter', 'MajorAxisLength', 'MinorAxisLength', 'Elongation', 'Compactness', 'Eccentricity', 'Solidity', 'Extent', 'Circularity', 'RadialDistanceSD', 'SecondMoment', 'Roughness', 'MinIntensity', 'MaxIntensity', 'MeanIntensity', 'SDIntensity', 'MinIntensityBG', 'MaxIntensityBG', 'MeanIntensityBG', 'SDIntensityBG', 'IntensityDifference', 'markov1', 'markov2', 'markov3', 'markov4', 'markov5', 'gabormean_0_0', 'gaborSD_0_0', 'gabormean_0_1', 'gaborSD_0_1', 'gabormean_0_2', 'gaborSD_0_2', 'gabormean_1_0', 'gaborSD_1_0', 'gabormean_1_1', 'gaborSD_1_1', 'gabormean_1_2', 'gaborSD_1_2', 'gabormean_2_0', 'gaborSD_2_0', 'gabormean_2_1', 'gaborSD_2_1', 'gabormean_2_2', 'gaborSD_2_2', 'gabormean_3_0', 'gaborSD_3_0', 'gabormean_3_1', 'gaborSD_3_1', 'gabormean_3_2', 'gaborSD_3_2', 'Contrast', 'Correlation', 'Energy', 'Homogeneity', 'Entropy', 'x_3rdordermoment', 'Inversevariance', 'Sumaverage', 'Variance', 'Clustertendency', 'MaxProbability']
    for i in range(1,5): #create trees with iterative ratings model  
        hed = copy.copy(hdr)
        print("Building tree with ", i, " radiologist ratings: ")
        print ("\n\n\n\n\n")
        tree = createTree(tf, hed, parents[i-1], children[i-1], 0, depth,i)
        if i == 4:
            print (tree)
        trees.append(tree)
    return trees  

def getTrees(head, t_features, np,nc,md,kround):
    param = [np,nc,md]
    with open("output/trees1.txt","r") as t:
        tree_file = t.read()
    trees = []
    clear = True
    if(len(tree_file)) != 0:
        trees = ast.literal_eval(tree_file)
        for i in range(3): 
            if trees[0][i] != param[i]: clear = True
    else:
        clear = True
    if clear:
        outputfile = "output/trees"+str(kround)+".txt"
        with open(outputfile,"wb") as t:
            trees.append(param)
            trees[1:] = buildTrees(head, t_features, np,nc,md)
            t.write(str(trees))
            t.close()
    return trees[1:]
#####################################
# MAIN SCRIPT
#####################################
f = open("output/BDT Output1.txt","w")
#treeFile = open("output/trees.txt","w")
LIDC_Data, header = importAllData("./data/modeBalanced/ModeBalanced_170_LIDC_809_Random.csv")
#importIdData("./data/clean/LIDC_809_Complete.csv")
#importData("./data/modeBalanced/small/Test_ModeBalanced_809.csv","./data/modeBalanced/small/Train_600_ModeBalanced_809_Random.csv","")
k_round = 1
k_best = [0,0.0,[],[],[],[],[],[]]
kfolds = 4
kf = KFold(len(LIDC_Data), kfolds)
test_header = copy.copy(header)
i = 0
for trn_ind, tst_ind in kf:
    header = copy.copy(test_header)
    trnLabels = []
    testLabels = []
    calibLabels = []
    trees = []
    train_features, calib_features = setTrain(LIDC_Data[trn_ind])
    test_features = LIDC_Data[tst_ind].tolist()
    # Console Output
    print("\n K-FOLD VALIDATION ROUND ",k_round," OF ",kfolds)
    k_round += 1 
    print("#################################")
    # Create Tree
    print ("Building Belief Decision Tree...") 
    nparent = 16
    nchild = 8
    maxdepth = 12
    print("CREATING BDT (d = ",maxdepth,", np = ",nparent,", nc = ",nchild,"): \n\n",file=f)
    # Create Tree
    trees = getTrees(header, train_features,nparent,nchild,maxdepth,k_round)
    ## Get actual data
    actualTrain = getPigns(train_features,4)
    actualTest = getPigns(test_features,4)
    actualCalib = getPigns(calib_features,4)
    actual_one = getPigns(test_features,1)
    actual_two = getPigns(test_features,2)
    actual_three = getPigns(test_features,3)
    # Classify calibration set & compute conformity
    print ("Classifying Calibration Set...") 
    calib_label_dct, calibLabels = classifyIterativeTrees(trees,test_header,calib_features)
    print(len(calibLabels))
    # Calibration Conformity
    print("Computing Calibration Conformity...")
    calib_conf = Train_Conformity(actualCalib, calibLabels)
    
#    print("\nCALIBRATION:", file=f)
#    for i in range (0,len(actualCalib)):
#        print("Probabilities: ", actualCalib[i], "\tConformity Score = ", calib_conf[i], file=f)
        
    train_label_dict, trnLabels = classifyIterativeTrees(trees,test_header,train_features)
    test_label_dict, tstLabels = classifyIterativeTrees(trees, test_header, test_features)
    tstLabels = test_label_dict[4]
    
#    selective classification methods    
#    modified_avg_test = getAverageBBA(test_label_dict)
#    modified_max_test = getMaxBBA(test_label_dict)
#    modified_con_test = getConfidentBBA(test_label_dict,calib_conf)
#    
#    selective_choice = selectiveChoice(test_label_dict,calib_conf)
#    selective_modal = getMode([modified_avg_test, modified_max_test, modified_con_test])
#    selective_avg = getAvy([modified_avg_test, modified_max_test, modified_con_test])

#    individual models
#    one = test_label_dict[1]
#    two = test_label_dict[2]
#    three = test_label_dict[3]
    four = test_label_dict[4]
    print(four, file=f)
    #classification for actual
    tst_proba = [getMax(case)+1 for case in actualTest]
    
    
    #evaluate classification accuracy
#    selective_choice_matrix = getConfusionMatrix(selective_avg, actualTest)
#    test_accy = getAccuracy(selective_choice_matrix, len(actualTest))
#    print("\nTESTING CONFUSION MATRIX SELECTIVE Classification BBA", file=f)
#    print(selective_choice_matrix,file=f)
#    print("\nTESTING ACCURACY = ", test_accy, file=f)
#        
#    
#    modal_avg_matrix = getConfusionMatrix(selective_avg, actualTest)
#    test_accy = getAccuracy(modal_avg_matrix, len(actualTest))
#    print("\nTESTING CONFUSION MATRIX SELECTIVE EVALUATION AVG BBA", file=f)
#    print(modal_avg_matrix,file=f)
#    print("\nTESTING ACCURACY = ", test_accy, file=f)
#    
#    modal_matrix = confusion_matrix(selective_modal,tst_proba)
#    test_accy = getAccuracy(modal_matrix, len(actualTest))
#    print("\nTESTING CONFUSION MATRIX SELECTIVE MODAL BBA", file=f)
#    print(modal_matrix,file=f)
#    print("\nTESTING ACCURACY = ", test_accy, file=f)
#    
#    test_class_matrix = getConfusionMatrix(modified_avg_test, actualTest)
#    test_accy = getAccuracy(test_class_matrix, len(tstLabels))
#    print("\nTESTING CONFUSION MATRIX AVG BBA", file=f)
#    print(test_class_matrix,file=f)
#    print("\nTESTING ACCURACY = ", test_accy, file=f)
#    
#    test_class_matrix = getConfusionMatrix(modified_max_test, actualTest)
#    test_accy = getAccuracy(test_class_matrix, len(tstLabels))
#    print("\nTESTING CONFUSION MATRIX MAX BBA", file=f)
#    print(test_class_matrix,file=f)
#    print("\nTESTING ACCURACY = ", test_accy, file=f)
#    
#    test_class_matrix = getConfusionMatrix(modified_con_test, actualTest)
#    test_accy = getAccuracy(test_class_matrix, len(tstLabels))
#    print("\nTESTING CONFUSION MATRIX BEST CONFIDENCE", file=f)
#    print(test_class_matrix,file=f)
#    print("\nTESTING ACCURACY = ", test_accy, file=f)
    
#    test_class_matrix = getConfusionMatrix(one, actual_one)
#    test_accy = getAccuracy(test_class_matrix, len(tstLabels))
#    print("\nTESTING CONFUSION MATRIX LABEL ONE", file=f)
#    print(test_class_matrix,file=f)
#    print("\nTESTING ACCURACY = ", test_accy, file=f)
#    
#    test_class_matrix = getConfusionMatrix(two, actual_two)
#    test_accy = getAccuracy(test_class_matrix, len(tstLabels))
#    print("\nTESTING CONFUSION MATRIX LABEL TWO", file=f)
#    print(test_class_matrix,file=f)
#    print("\nTESTING ACCURACY = ", test_accy, file=f)
#    
#    test_class_matrix = getConfusionMatrix(three, actual_three)
#    test_accy = getAccuracy(test_class_matrix, len(tstLabels))
#    print("\nTESTING CONFUSION MATRIX THREE LABELS", file=f)
#    print(test_class_matrix,file=f)
#    print("\nTESTING ACCURACY = ", test_accy, file=f)
##    
    test_class_matrix = getConfusionMatrix(four, actualTest)
    test_accy = getAccuracy(test_class_matrix)
    print("\nTESTING CONFUSION MATRIX LABEL FOUR", file=f)
    print(test_class_matrix,file=f)
    print("\nTESTING ACCURACY = ", test_accy, file=f)
    
    test_conf = Test_Conformity(four)
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
    writeData("Training", trainhead, "output/TrainOutput.csv", "wb", actualTrain, trnLabels, confidence, credibility, 0)
    writeData("Testing", testhead, "output/TestOutput.csv", "wb", actualTest, four , confidence, credibility, len(trnLabels))
f.close()
print("Done")