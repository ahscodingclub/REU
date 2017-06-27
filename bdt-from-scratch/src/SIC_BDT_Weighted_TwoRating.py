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

        pign = five_to_two(pign)
        nodePigns.append(pign) # Add pign to list of pigns
    return nodePigns
    
def five_to_two(lst):
    theta = lst[2]/2
    pb = lst[0] + (lst[1] * 0.75) + theta + (lst[3] * 0.25)
    pm = lst[4] + (lst[3] * 0.75) + theta + (lst[1] * 0.25)
    return [pb,pm]
    
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
    
    return bestFeature, bestValue, bestGain, baseApign, basePigns

# DETERMINE IF ALL LIST ITEMS ARE THE SAME
def all_same(items):
    return all(x == items[0] for x in items)


def removeColumn(dataset, column):
    cases = []
    for case in dataset:
        cases.append(case[:column] + case[column+1:])
    return cases
        
# CREATING THE TREE RECURSIVELY
def createTree(dataset, labels, min_parent, min_child, curr_depth, max_depth,ratings):

    # If labels exist in this subset, determine best split
    if len(labels) >= 1:
        
        # Get Best Split Information
        output = bestFeatureSplit(dataset, len(labels), min_parent, min_child,ratings) # index of best feature
        bestFeat = output[0]
        bestVal = output[1]
        bestGainRatio = output[2]
        baseApign = output[3]
        basePigns = output[4]
        
        # Get label of best feature
        bestFeatLabel = labels[bestFeat]
        
        # Create root node
        decision_tree = {bestFeatLabel:{"BBA":baseApign}}
        # Stopping Conditions
        if (bestGainRatio == 0) and (bestFeat == 0) and (bestVal == 0):
            decision_tree = {"Leaf":{"BBA":baseApign}}
            return decision_tree, labels
        elif (bestGainRatio == 0):
            decision_tree = {"Leaf":{"BBA":baseApign}}
            return decision_tree, labels
        elif (all_same(basePigns)):
            decision_tree = {"Leaf":{"BBA":baseApign}}
            return decision_tree, labels
        elif (curr_depth == max_depth): 
            decision_tree = {"Leaf":{"BBA":baseApign}}
            return decision_tree, labels
            
        # Recursive Call
        else:
            print ("   Splitting at depth ", curr_depth, "...")  
       
            del(labels[bestFeat]) #remove chosen label from list of labels
            # Create a split
            decision_tree[bestFeatLabel][bestVal] = {}            
            subLabels = copy.copy(labels)
            low_Set, high_Set = splitData(dataset, bestFeat, bestVal) # returns low & high set
            lowSet = removeColumn(low_Set,bestFeat)
            highSet = removeColumn(high_Set,bestFeat)

            decision_tree[bestFeatLabel][bestVal]["left"], subLabels = createTree(lowSet, subLabels, min_parent, min_child,\
                                                                       curr_depth+1, max_depth,ratings) 
            i = len(labels)-1     
            j = len(subLabels)-1
            while i >= 0:
                if labels[i] != subLabels[j]:
                    highSet = removeColumn(highSet,i)
                    i -= 1
                else: 
                    i -= 1
                    j -= 1
            
            labels = copy.copy(subLabels)   
            decision_tree[bestFeatLabel][bestVal]["right"], subLabels = createTree(highSet, subLabels, min_parent, min_child,\
                                                                                     curr_depth+1, max_depth,ratings)
            return decision_tree, subLabels
    
    # If no labels left, then STOP
    elif (len(labels) < 1):
        return

#####################################
# OUTPUT RESULTS DATAFILES
#####################################
def writeData(trainOrTest, header, filename, params, actual, predicted, conf, cred, id_start):
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
        print (predicted)
        for i in range(0, len(predicted)):
            rmse = 0            
            r = 0
            roc = 0
            # Calculate RMSE and Pearson Correlation
            for j in range(2):
                rmse += pow((actual[i][j] - predicted[i][j]), 2)
            rmse /= ratings
            rmse = math.sqrt(rmse)
            
            r = 0.0
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
    cos_sim, similarities, ratings = CosineSimilarity(actual, predicted)

    # Output to Console
    print("\n" + trainOrTest + "\n" + str(conf_matrix))
    print(trainOrTest,"Accuracy = ", '{:.4}'.format(accy * 100), "%")

    
    # Output Confusion Matrices, Accuracies, AUCdt, and ROC AUC
    print("\n", trainOrTest, "Confusion Matrix", file=f)
    print(conf_matrix,file=f)
    np.set_printoptions(precision=3)
    print("Accuracy = ", '{:.4}'.format(accy * 100), "%", file=f)
    
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

def PValue2(test, calib):
    pvalues = []
    for i in range(len(test)): # for each case in testing 
        counts = [0,0] #stores count of test conformity greater than equal to calib 
        for j in range(0,2): #for each label in test
            val = test[i][j] #conformity value of label
            for k in range(0,len(calib[0])): #for each conformity value 
                if calib[i][k] <= val: #if less than val increase count of label
                    counts[j] += 1
        pvalues.append([count/float(len(calib[0])+1) for count in counts]) #append p values for case
    return pvalues   

#####################################
# CONFUSION MATRIX BUILDING
#####################################
def getConfusionMatrix(predicted,actual): #returns misclassification matrix
    # TEST: Find greatest probability to use as prediction
    pred_proba = [getMax(case)+1 for case in predicted]
    act_proba = [getMax(case)+1 for case in actual]
    return confusion_matrix(act_proba,pred_proba)
    
def getConfusionMod(predicted,actual, conf): #returns misclassification matrix
    # TEST: Find greatest probability to use as prediction
    pred_proba = [getMax(case)+1 for case in predicted]
    act_proba = [getMax(case)+1 for case in actual]
    
    avg_incorrect = 0.0
    count = 0
    for i in range(len(actual)):
        if pred_proba[i] == act_proba[i]:
            count += 1
            avg_incorrect += conf[i]
            #print(max(predicted[i]), max(actual[i]), conf[i])
    #print ("avg incorrect proba ", avg_incorrect/count)

#writes data of cases classified correctly    
def getCorrectData(predicted,actual, conf): #returns misclassification matrix
    # TEST: Find greatest probability to use as prediction
    pred_proba = [getMax(case)+1 for case in predicted]
    act_proba = [getMax(case)+1 for case in actual]
    with open("../output/CorrectCases.csv","wb") as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Actual1','Actual2','Predicted1','Predicted2','Confidence','RMSE'])
        
        for i in range(0, len(predicted)):
            rmse = 0    
            
            for j in range(2):
                rmse += pow((actual[i][j] - predicted[i][j]), 2)
            rmse /= 2.0
            rmse = math.sqrt(rmse)
            if(pred_proba[i] == act_proba[i]):
                writer.writerow([
                                     actual[i][0], actual[i][1],\
                                     predicted[i][0], predicted[i][1],\
                                     conf[i],rmse])
                                     
def getAccuracy(class_matrix): # returns accuracy of misclassification matrix 
    accy = 0.0
    for j in range(2):
        accy += class_matrix[j][j]
    print("accy: \n", accy)
    mat_sum = sum2D(class_matrix)
    print("sum: \n" mat_sum)
            
    return accy / mat_sum  
    
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
    for i in range(0,len(lst)):
        m = getMax(lst[i])
        conf = 1 - max([x for k,x in enumerate(lst[i]) if k != m])
        if conf >= .95:
            return i
            
    return -1
      
# function adds weight to a bba **unused**
def getMaxBBA2(bbas):
    for k,bba in enumerate(bbas):
        temp_max = max(bba)
        if temp_max >= .7:
            return k
    return -1
    
#selective iterative classification method. 
#selects model3 based on confidence, max bba value in a vector, and averages the bbas if no model is satisfactory.
def selectiveChoice(label_dict,calib):
    modified_labels = []
    avg_chosen = 0
    max_chosen = 0
    confidence_chosen = 0
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
        p_values = PValue2(conform, calib)  
        ind = highestConfidence(p_values) #if higher than .9 return index of bba vector
        ind2 = getMaxBBA2(bbas)
        
        if ind2 != -1:
            max_chosen += 1
            used_trees[ind2] += 1
            choice = bbas[ind2]
        elif ind != -1: #no max and confidence is chosen
            if(max(bbas[ind]) > max(avg_bba)):
                confidence_chosen += 1
                used_trees[ind] += 1
                choice = bbas[ind]
            else:
                choice = avg_bba
                avg_chosen += 1
                used_trees[-1] += 1
        else: #compute max probability
            avg_chosen += 1
            used_trees[-1] += 1
            choice = avg_bba
            
        modified_labels.append(choice)
    return modified_labels    

#selective iterative method that chooses  model based on confidence of the bba            
def getConfidentBBA(train_label_dict, calib):
    modified_train = []
    used_trees = [0,0,0,0,0]
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
            used_trees[-1] += 1
            modified_train.append(avg_bba)
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
    tf = train_features
    hdr = ['Area', 'ConvexArea', 'Perimeter', 'ConvexPerimeter', 'EquivDiameter', 'MajorAxisLength', 'MinorAxisLength', 'Elongation', 'Compactness', 'Eccentricity', 'Solidity', 'Extent', 'Circularity', 'RadialDistanceSD', 'SecondMoment', 'Roughness', 'MinIntensity', 'MaxIntensity', 'MeanIntensity', 'SDIntensity', 'MinIntensityBG', 'MaxIntensityBG', 'MeanIntensityBG', 'SDIntensityBG', 'IntensityDifference', 'markov1', 'markov2', 'markov3', 'markov4', 'markov5', 'gabormean_0_0', 'gaborSD_0_0', 'gabormean_0_1', 'gaborSD_0_1', 'gabormean_0_2', 'gaborSD_0_2', 'gabormean_1_0', 'gaborSD_1_0', 'gabormean_1_1', 'gaborSD_1_1', 'gabormean_1_2', 'gaborSD_1_2', 'gabormean_2_0', 'gaborSD_2_0', 'gabormean_2_1', 'gaborSD_2_1', 'gabormean_2_2', 'gaborSD_2_2', 'gabormean_3_0', 'gaborSD_3_0', 'gabormean_3_1', 'gaborSD_3_1', 'gabormean_3_2', 'gaborSD_3_2', 'Contrast', 'Correlation', 'Energy', 'Homogeneity', 'Entropy', 'x_3rdordermoment', 'Inversevariance', 'Sumaverage', 'Variance', 'Clustertendency', 'MaxProbability']
    for i in range(1,5): #create trees with iterative ratings model  
        hed = copy.copy(hdr)
        print("Building tree with ", i, " radiologist ratings: ")
        print (len(hed))
        tree = createTree(tf, hed, parent, child, 0, depth,i)
        tree = tree[0]
        trees.append(tree)
    return trees  

def getTrees(head, t_features, np,nc,md,kround):
    param = [np,nc,md]
    trees = []
    outputfile = "../output/tree.txt"
    with open(outputfile,"wb") as t:
        trees.append(param)
        trees[1:] = buildTrees(head, t_features, np,nc,md)
        t.write(str(trees))
        t.close()
    return trees[1:]

#####################################
# MAIN SCRIPT
#####################################
pignType = input("Pignistic Type?\n1.Mean\n2.Median\n3.Mode\n4.Distribution\ntype: ")

f = raw_input("file for confusion matrix: ") 
f = open(f, "w")
#treeFile = open("output/trees.txt","w")
LIDC_Data, header = importAllData("../data/modeBalanced/ModeBalanced_170_LIDC_809_Random.csv")
#importIdData("./data/clean/LIDC_809_Complete.csv")
#importData("./data/modeBalanced/small/Test_ModeBalanced_809.csv","./data/modeBalanced/small/Train_600_ModeBalanced_809_Random.csv","")
k_accy = 0.0
k_round = 1
k_best = [0,0.0,[],[],[],[],[],[]]
#kfolds = 4
kfolds = 2
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
    print("#################################")
    # Create Tree
    print ("Building Belief Decision Tree...") 
#   nparent = 28
    nparent = 8
#   nchild = 14
    nchild = 4
#   maxdepth = 12
    maxdepth = 2
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
    # Calibration Conformity
    print("Computing Calibration Conformity...")
    calibs = []
    for i in range(4):
        calibs.append(Train_Conformity(actualCalib, calib_label_dct[i+1]))
    
    train_label_dict, trnLabels = classifyIterativeTrees(trees,test_header,train_features)
    test_label_dict, tstLabels = classifyIterativeTrees(trees, test_header, test_features)
    tstLabels = test_label_dict[4]
    
    #    selective classification methods    
    modified_avg_test = getAverageBBA(test_label_dict)
    modified_max_test = getMaxBBA(test_label_dict)
    modified_con_test = getConfidentBBA(test_label_dict,calibs)
    
    selective_choice = selectiveChoice(test_label_dict,calibs)
    selective_modal = getMode([modified_avg_test, modified_max_test, modified_con_test])
    selective_avg = getAvy([modified_avg_test, modified_max_test, modified_con_test])
    
    #    individual models
    one = test_label_dict[1]
    two = test_label_dict[2]
    three = test_label_dict[3]
    four = test_label_dict[4]
    #classification for actual
    tst_proba = [getMax(case)+1 for case in actualTest]
    
    
    #evaluate classification accuracy
    selective_choice_matrix = getConfusionMatrix(selective_choice, actualTest)
    choice_accy = getAccuracy(selective_choice_matrix)
    
    #print("\nTESTING CONFUSION MATRIX SELECTIVE Classification BBA", file=f)
    #print(selective_choice_matrix,file=f)
    #print("\nTESTING ACCURACY = ", choice_accy, file=f)
        
    
    modal_avg_matrix = getConfusionMatrix(selective_avg, actualTest)
    test_accy = getAccuracy(modal_avg_matrix)
    #print("\nTESTING CONFUSION MATRIX SELECTIVE EVALUATION AVG BBA", file=f)
    #print(modal_avg_matrix,file=f)
    #print("\nTESTING ACCURACY = ", test_accy, file=f)
    
    modal_matrix = confusion_matrix(selective_modal,tst_proba)
    test_accy = getAccuracy(modal_matrix)
    #print("\nTESTING CONFUSION MATRIX SELECTIVE MODAL BBA", file=f)
    #print(modal_matrix,file=f)
    #print("\nTESTING ACCURACY = ", test_accy, file=f)
    
    test_class_matrix = getConfusionMatrix(modified_avg_test, actualTest)
    test_accy = getAccuracy(test_class_matrix)
    #print("\nTESTING CONFUSION MATRIX AVG BBA", file=f)
    #print(test_class_matrix,file=f)
    #print("\nTESTING ACCURACY = ", test_accy, file=f)
    
    test_class_matrix = getConfusionMatrix(modified_max_test, actualTest)
    test_accy = getAccuracy(test_class_matrix)
    #print("\nTESTING CONFUSION MATRIX MAX BBA", file=f)
    #print(test_class_matrix,file=f)
    #print("\nTESTING ACCURACY = ", test_accy, file=f)
    
    test_class_matrix = getConfusionMatrix(modified_con_test, actualTest)
    test_accy = getAccuracy(test_class_matrix)
    #print("\nTESTING CONFUSION MATRIX BEST CONFIDENCE", file=f)
    #print(test_class_matrix,file=f)
    #print("\nTESTING ACCURACY = ", test_accy, file=f)
    
    test_class_matrix = getConfusionMatrix(one, actual_one)
    test_accy = getAccuracy(test_class_matrix)
#    print("\nTESTING CONFUSION MATRIX LABEL ONE", file=f)
#    print(test_class_matrix,file=f)
#    print("\nTESTING ACCURACY = ", test_accy, file=f)
    
    test_class_matrix = getConfusionMatrix(two, actual_two)
    test_accy = getAccuracy(test_class_matrix)
#    print("\nTESTING CONFUSION MATRIX LABEL TWO", file=f)
#    print(test_class_matrix,file=f)
#    print("\nTESTING ACCURACY = ", test_accy, file=f)
    
    test_class_matrix = getConfusionMatrix(three, actual_three)
    test_accy = getAccuracy(test_class_matrix)
#    print("\nTESTING CONFUSION MATRIX THREE LABELS", file=f)
#    print(test_class_matrix,file=f)
#    print("\nTESTING ACCURACY = ", test_accy, file=f)
    ##    
    test_class_matrix = getConfusionMatrix(four, actualTest)
    test_accy = getAccuracy(test_class_matrix)
#    print("\nTESTING CONFUSION MATRIX LABEL FOUR", file=f)
#    print(test_class_matrix,file=f)
#    print("\nTESTING ACCURACY = ", test_accy, file=f)
    
    test_conf = Test_Conformity(modified_max_test)
    p_vals = PValue(test_conf,calibs[3])
    confidence = []
    credibility = []

    for i, val in enumerate(p_vals):
        m = val.index(max(val))
        sec = max([x for k,x in enumerate(val) if k != m])
    #    print(i, "  P-values: ", val, "  Class Label: ", val.index(max(val))+1, "  Confidence: ", 1-sec, "\tCredibility: ", max(val))
        confidence.append(1 - sec)
        credibility.append(max(val))
    #    test_class_matrix = getConfusionMod(four, actualTest, confidence)    
    if k_round == 1:
        k_accy = choice_accy
        k_best = [k_round, k_accy, actualTrain, trnLabels, actualTest, selective_choice, confidence, credibility]
        best_tree = trees
        # Output data to csv and text files
        accuracy = k_best[1]
        actualTrain = k_best[2]
        trainLabels = k_best[3]
        actualTest = k_best[4]
        testLabel = k_best[5]
        confidence = k_best[6]
        credibility = k_best[7]  
        trainhead = "Training Data: (d = " + str(maxdepth) + " | np = " + str(nparent) + " | nc = " + str(nchild) + ")"
        testhead = "Testing Data: (d = " + str(maxdepth) + " | np = " + str(nparent) + " | nc = " + str(nchild) + ")"
        #("Training", trainhead, "output/TrainOutput.csv", "wb", actualTrain, trnLabels, confidence, credibility, 0)
        writeData("Testing", testhead, "../output/TestOutput.csv", "wb", actualTest, testLabel , confidence, credibility, len(trnLabels))
        getCorrectData(modified_max_test, actualTest, confidence)
    
    k_round += 1
   
writeData("Training", trainhead, "../output/TrainOutput.csv", "wb", actualTrain, trnLabels, confidence, credibility, 0)
writeData("Testing", testhead, "../output/TestOutput1.csv", "wb", actualTest, selective_avg , confidence, credibility, len(trnLabels))

#print("Best Fold Tree: ", file=f)
#print(best_tree, file=f)
f.close()
print("Done")
