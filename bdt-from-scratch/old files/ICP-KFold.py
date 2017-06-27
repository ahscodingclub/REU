"""
#####################################
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
import csv # Read and write csv files
import copy
from sklearn.metrics import confusion_matrix # Assess misclassification
from sklearn.metrics import roc_auc_score # Get ROC area under curve
#from pyemd import emd # Earth mover's distance
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
    
# SET TRAINING DATA
def setTrain(trn_data_array):
    global train_features
    global calib_features
    #rand_trn = [ trn_data_array[i].tolist() for i in random.sample(xrange(len(trn_data_array)), len(trn_data_array)) ]
    #np.random.shuffle(trn_data_array)
    split = int((6.0/7.0) * len(trn_data_array))
    #print(split)
    train_features = trn_data_array[:split].tolist() # training data
    #calib_features = [ train_features[i] for i in sorted(random.sample(xrange(len(train_features)), split)) ]
    calib_features = trn_data_array[split:].tolist() # calibration data
    
# SET TESTING DATA
#def setTest():
#    global test_features
#    test_features = tst_data_array[:].tolist() # test pixel area
    
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
    #print (nodePigns)
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
            print ("   Splitting at depth ", curr_depth, "...")  
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
# CONFORMAL PREDICTION
#####################################

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
        actual_class = getMax(actual[i]) #maximum value of predicted
        #max_ind = actual[i].index(max(actual[i])) #index of max actual
        max_ind = actual_class        
        pred = predicted[i][max_ind] #probability of predicted chosen predicted label
        max_val = max([prob for k,prob in enumerate(predicted[i]) if k != max_ind]) #max value in subset without chosen label
        conform.append(pred - max_val) #append conformity score
    return conform

def Test_Conformity(predicted):
    conform = []
    for i in range(len(predicted)): # for each case
        conf_score = []
        for j in range(len(predicted[i])): # for each rating
            cls = predicted[i][j]  # chosen class label
            max_val = max([prob for k,prob in enumerate(predicted[i]) if k != j]) # max val of remaining class labels
            conf_score.append(cls-max_val) # chosen - max remaining value
        conform.append(conf_score) # append conformity score
    return conform
    
def PValue(test, calib):
    pvalues = []
    for i in range(len(test)): # for each case in testing data
        counts = [0,0,0,0,0] # stores count of test conformity greater than equal to calib 
        for j in range(0,5): # for each label in test
            val = test[i][j] # testing conformity score of label
            for k in range(0,len(calib)): # for each calibration conformity value 
                if calib[k] <= val: # if less than val increase count of this label
                    counts[j] += 1
        pvalues.append([count/float(len(calib)+1) for count in counts]) #append p values for case
    return pvalues

#####################################
# EVALUATION METHODS
#####################################

# Calculate a confusion matrix given predicted and actual
def getConfusionMatrix(predicted, actual): #returns misclassification matrix
    # TEST: Find greatest probability to use as prediction
    pred_proba = [getMax(case)+1 for case in predicted]
    act_proba =  [getMax(case)+1 for case in actual]
    return confusion_matrix(act_proba,pred_proba) # Top axis is predicted, side axis is actual

# Calculate accuracy given a confusion matrix
def getAccuracy(class_matrix, num_cases): # returns accuracy of misclassification matrix 
    accy = 0.0
    for j in range(0,5):
        accy += class_matrix[j][j]
    return accy / num_cases

# Sum elements in a 2D array
def sum2D(input):
    return sum(map(sum, input))

# MODIFIED ACCURACY EQUATION
def modConfusion(pred, actual):
    matrix = [[0,0,0,0,0],
              [0,0,0,0,0],
              [0,0,0,0,0],
              [0,0,0,0,0],
              [0,0,0,0,0]]
    
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
                
    #print (matrix)
    return matrix

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
        #print(ratings)
    return (average/len(predicted), similarities, ratings)

# Area Under ROC Curve for a single case
# predicted = [0, .2, .4, .4, 0]
# actual = [0, .25, .5, .25, 0]
def getROC(predicted, actual):
    binaryActual = [0, 0, 0, 0, 0]
    maxIndex = getMax(actual)
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
    
#####################################
# OUTPUT RESULTS DATAFILES
#####################################
def writeData(trainOrTest, header, filename, params, actual, predicted, conf, cred, id_start):
    avgROC = 0    
    with open(filename, params) as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if trainOrTest == "Training":
            writer.writerow(['Nodule ID',\
                             'Actual [1]',      'Actual [2]',       'Actual [3]',       'Actual [4]',       'Actual [5]',\
                             'Predicted [1]' ,  'Predicted [2]',    'Predicted [3]',    'Predicted [4]',    'Predicted [5]',\
                             'RMSE', 'Pearson r', 'ROC AUC', header])
        else:
            writer.writerow(['Nodule ID',\
                             'Actual [1]',      'Actual [2]',       'Actual [3]',       'Actual [4]',       'Actual [5]',\
                             'Predicted [1]' ,  'Predicted [2]',    'Predicted [3]',    'Predicted [4]',    'Predicted [5]',\
                             'RMSE', 'Pearson r', 'ROC AUC', 'Confidence', 'Credibility', header])
        
        # For each case at this node
        print(len(actual),len(predicted))
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
            if trainOrTest == "Training":
                writer.writerow([set_data_array[i+id_start][2],\
                                 actual[i][0], actual[i][1], actual[i][2], actual[i][3], actual[i][4],\
                                 predicted[i][0], predicted[i][1], predicted[i][2], predicted[i][3], predicted[i][4],\
                                 rmse, r, roc])
            else:
                writer.writerow([set_data_array[i+id_start][2],\
                                 actual[i][0], actual[i][1], actual[i][2], actual[i][3], actual[i][4],\
                                 predicted[i][0], predicted[i][1], predicted[i][2], predicted[i][3], predicted[i][4],\
                                 rmse, r, roc, conf[i], cred[i]])
            
    # Calculate Accuracy and AUCdt
    conf_matrix = getConfusionMatrix(predicted, actual)
    print(conf_matrix)
    accy = getAccuracy(conf_matrix, len(predicted))
    cMatrix = modConfusion(predicted, actual)
    modAccy = getAccuracy(cMatrix, len(predicted))
    myAUCdt = scipy.integrate.quad(AUCdt, 0, 1, args = (actual, predicted))
    avgROC /= len(predicted)
    cos_sim, similarities, ratings = CosineSimilarity(actual, predicted)
    
    # Output Confusion Matrices, Accuracies, AUCdt, and ROC AUC
    print("\n", trainOrTest, "Confusion Matrix", file=f)
    print(conf_matrix,file=f)
    print("\n", trainOrTest, "MOD Confusion Matrix", file=f)
    np.set_printoptions(precision=3)
    print("",np.asarray(cMatrix[0]),"\n",np.asarray(cMatrix[1]),"\n",np.asarray(cMatrix[2]),"\n",np.asarray(cMatrix[3]),\
        "\n",np.asarray(cMatrix[4]),file=f)
    print(" Matrix Sum = ",sum2D(cMatrix),"\n",file=f)
    print("Accuracy = ", '{:.4}'.format(accy * 100), "%", file=f)
    print("MOD Accuracy = ", '{:.4}'.format(modAccy * 100), "%", file=f)
    print("AUCdt JD = ", '{:.4}'.format(myAUCdt[0]), " with error of ", '{:.4}'.format(myAUCdt[1]), file=f)
    print("Avg ROC AUC = ", '{:.4}'.format(avgROC), file=f)
    #print("Average Cosine Similarity: ", cos_sim, file=f)
    #print("Cosine Counts: ", file=f)
    #for k in ratings.keys(): 
    #    print("("+str(k-0.25) + " - " + str(k)+")", ratings[k],file=f)

#####################################
# DRAW THE DECISION TREE
#####################################

# IF tree in file is same tree we want, USE IT rather than building it
def getTrees(tf,head,np,nc,mind,md, switch):
    param = [np,nc,md]
    if(switch):
         with open("output/tree.txt","wb") as t:
            trees = [param,createTree(train_features, header, nparent, nchild, mind, maxdepth)]
            t.write(trees.__repr__())
            return trees[1]
    else:
        with open("output/tree.txt","r") as t:
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
            with open("output/tree.txt","wb") as t:
                trees = [param,createTree(train_features, header, nparent, nchild, mind, maxdepth)]
                t.write(trees.__repr__())
        return trees[1]
    
def draw(parent_name, child_name):
    global graph
    edge = pydot.Edge(parent_name, child_name)
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

#####################################
# MAIN SCRIPT: Build, Classify, Output
#####################################       
# Setup
f = open("output/BDT Output.txt","w")

importIdData("./data/clean/LIDC_809_Complete.csv")
importAllData("./data/modeBalanced/ModeBalanced_170_LIDC_809_Random.csv")
#importData("./data/modeBalanced/Test_ModeBalanced_809.csv","./data/modeBalanced/Train_600_ModeBalanced_809_Random.csv","")
#importData("./data/meanBalanced/Balanced(40)_Clean_809_Test.csv","./data/meanBalanced/Balanced(120)_Clean_809_Train_Random.csv","")

#setTrain()
#setTest()
test_header = copy.copy(header)
######K-FOLD VALIDATION######
kf = KFold(len(LIDC_Data), 4)
k_round = 1
for trn_ind, tst_ind in kf:
    header = copy.copy(test_header)
    trainLabels = []
    testLabels = []
    calibLabels = []
    setTrain(LIDC_Data[trn_ind])
    test_features = LIDC_Data[tst_ind].tolist()
    print (trn_ind, tst_ind)
    print("Train Size: ", len(train_features))
    print("Test Size: ", len(test_features))
    # Create Tree
    print("\n K-FOLD VALIDATION ROUND ",k_round,file=f )
    print("########################",file=f)
    print ("Building Belief Decision Tree...") 
    
    nparent = 20
    nchild = 10
    maxdepth = 25
    
    print("CREATING BDT (d = ",maxdepth,", np = ",nparent,", nc = ",nchild,"): \n\n",file=f)
    tree = getTrees(train_features,header,nparent,nchild,0,maxdepth, True) #setting "switch = False" will make new tree each time
    #tree = createTree(train_features, header, nparent, nchild, 0, maxdepth)
    #tree = {'Elongation': {1.238539467: {'right': {'Compactness': {1.3985949930000001: {'right': {'gabormean_1_1': {106.557: {'right': {'MeanIntensityBG': {'BBA': [0.12333333333333334, 0.20666666666666667, 0.36333333333333334, 0.15666666666666668, 0.15], 406.2409: {'right': {'Leaf': {'BBA': [0.725, 0.05, 0.075, 0.05, 0.1]}}, 'left': {'Area': {480.0: {'right': {'Leaf': {'BBA': [0.0, 0.0, 0.125, 0.125, 0.75]}}, 'left': {'gaborSD_2_2': {62.6166: {'right': {'Leaf': {'BBA': [0.1875, 0.25, 0.53125, 0.0, 0.03125]}}, 'left': {'ConvexPerimeter': {39.79898987: {'right': {'markov3': {600.0549: {'right': {'gaborSD_0_1': {602.5217: {'right': {'Leaf': {'BBA': [0.0, 0.25, 0.625, 0.09375, 0.03125]}}, 'left': {'Variance': {61.1378: {'right': {'gabormean_3_2': {51.7785: {'right': {'Leaf': {'BBA': [0.0, 0.09375, 0.53125, 0.34375, 0.03125]}}, 'left': {'Leaf': {'BBA': [0.0, 0.3055555555555556, 0.19444444444444445, 0.3888888888888889, 0.1111111111111111]}}}, 'BBA': [0.0, 0.20588235294117646, 0.35294117647058826, 0.36764705882352944, 0.07352941176470588]}}, 'left': {'Leaf': {'BBA': [0.0625, 0.09375, 0.21875, 0.3125, 0.3125]}}}, 'BBA': [0.02, 0.17, 0.31, 0.35, 0.15]}}}, 'BBA': [0.015151515151515152, 0.1893939393939394, 0.38636363636363635, 0.2878787878787879, 0.12121212121212122]}}, 'left': {'Leaf': {'BBA': [0.0, 0.40625, 0.5, 0.09375, 0.0]}}}, 'BBA': [0.012195121951219513, 0.23170731707317074, 0.40853658536585363, 0.25, 0.0975609756097561]}}, 'left': {'Leaf': {'BBA': [0.0, 0.4375, 0.5625, 0.0, 0.0]}}}, 'BBA': [0.01020408163265306, 0.2653061224489796, 0.4336734693877551, 0.20918367346938777, 0.08163265306122448]}}}, 'BBA': [0.03508771929824561, 0.2631578947368421, 0.4473684210526316, 0.17982456140350878, 0.07456140350877193]}}}, 'BBA': [0.03076923076923077, 0.23076923076923078, 0.4076923076923077, 0.17307692307692307, 0.1576923076923077]}}}}}, 'left': {'MaxIntensityBG': {499.5232: {'right': {'Leaf': {'BBA': [0.875, 0.05, 0.0, 0.05, 0.025]}}, 'left': {'gabormean_3_1': {74.8974: {'right': {'Leaf': {'BBA': [0.1875, 0.4375, 0.28125, 0.09375, 0.0]}}, 'left': {'gaborSD_1_0': {70.7685: {'right': {'Leaf': {'BBA': [0.03125, 0.28125, 0.65625, 0.03125, 0.0]}}, 'left': {'Perimeter': {48.87005769: {'right': {'EquivDiameter': {59.35533906: {'right': {'ConvexPerimeter': {267.0: {'right': {'RadialDistanceSD': {32.05059773: {'right': {'gaborSD_3_0': {90.9131: {'right': {'Leaf': {'BBA': [0.0, 0.03125, 0.3125, 0.21875, 0.4375]}}, 'left': {'Leaf': {'BBA': [0.03125, 0.1875, 0.09375, 0.3125, 0.375]}}}, 'BBA': [0.015625, 0.109375, 0.203125, 0.265625, 0.40625]}}, 'left': {'Leaf': {'BBA': [0.03125, 0.0, 0.375, 0.34375, 0.25]}}}, 'BBA': [0.020833333333333332, 0.07291666666666667, 0.2604166666666667, 0.2916666666666667, 0.3541666666666667]}}, 'left': {'Leaf': {'BBA': [0.0, 0.1875, 0.28125, 0.46875, 0.0625]}}}, 'BBA': [0.015625, 0.1015625, 0.265625, 0.3359375, 0.28125]}}, 'left': {'Leaf': {'BBA': [0.125, 0.15625, 0.375, 0.34375, 0.0]}}}, 'BBA': [0.0375, 0.1125, 0.2875, 0.3375, 0.225]}}, 'left': {'Leaf': {'BBA': [0.0, 0.3125, 0.53125, 0.15625, 0.0]}}}, 'BBA': [0.03125, 0.14583333333333334, 0.328125, 0.3072916666666667, 0.1875]}}}, 'BBA': [0.03125, 0.16517857142857142, 0.375, 0.26785714285714285, 0.16071428571428573]}}}, 'BBA': [0.05078125, 0.19921875, 0.36328125, 0.24609375, 0.140625]}}}, 'BBA': [0.16216216216216217, 0.17905405405405406, 0.3141891891891892, 0.2195945945945946, 0.125]}}}, 'BBA': [0.14261744966442952, 0.1929530201342282, 0.3389261744966443, 0.18791946308724833, 0.13758389261744966]}}, 'left': {'markov2': {168.9411: {'right': {'MeanIntensityBG': {481.3424: {'right': {'Leaf': {'BBA': [0.7083333333333334, 0.08333333333333333, 0.20833333333333334, 0.0, 0.0]}}, 'left': {'gaborSD_3_2': {86.5451: {'right': {'Leaf': {'BBA': [0.0, 0.0, 0.34375, 0.15625, 0.5]}}, 'left': {'Entropy': {62.2277: {'right': {'Leaf': {'BBA': [0.1875, 0.375, 0.375, 0.03125, 0.03125]}}, 'left': {'gaborSD_2_0': {45.9409: {'right': {'Clustertendency': {64.15: {'right': {'Leaf': {'BBA': [0.03125, 0.28125, 0.625, 0.03125, 0.03125]}}, 'left': {'SecondMoment': {0.7679738559999999: {'right': {'Leaf': {'BBA': [0.03125, 0.25, 0.53125, 0.09375, 0.09375]}}, 'left': {'Correlation': {56.5184: {'right': {'Leaf': {'BBA': [0.0, 0.21875, 0.53125, 0.25, 0.0]}}, 'left': {'Leaf': {'BBA': [0.0, 0.11666666666666667, 0.3, 0.4166666666666667, 0.16666666666666666]}}}, 'BBA': [0.0, 0.15217391304347827, 0.3804347826086957, 0.358695652173913, 0.10869565217391304]}}}, 'BBA': [0.008064516129032258, 0.1774193548387097, 0.41935483870967744, 0.2903225806451613, 0.10483870967741936]}}}, 'BBA': [0.01282051282051282, 0.1987179487179487, 0.46153846153846156, 0.23717948717948717, 0.08974358974358974]}}, 'left': {'Leaf': {'BBA': [0.0625, 0.15625, 0.125, 0.3125, 0.34375]}}}, 'BBA': [0.02127659574468085, 0.19148936170212766, 0.40425531914893614, 0.25, 0.13297872340425532]}}}, 'BBA': [0.045454545454545456, 0.21818181818181817, 0.4, 0.21818181818181817, 0.11818181818181818]}}}, 'BBA': [0.03968253968253968, 0.19047619047619047, 0.39285714285714285, 0.21031746031746032, 0.16666666666666666]}}}, 'BBA': [0.14666666666666667, 0.17333333333333334, 0.36333333333333334, 0.17666666666666667, 0.14]}}, 'left': {'SDIntensity': {1412.0: {'right': {'Leaf': {'BBA': [0.75, 0.05555555555555555, 0.19444444444444445, 0.0, 0.0]}}, 'left': {'Inversevariance': {0.7929999999999999: {'right': {'Leaf': {'BBA': [0.0, 0.057692307692307696, 0.21153846153846154, 0.19230769230769232, 0.5384615384615384]}}, 'left': {'Variance': {0.0022: {'right': {'MaxIntensityBG': {1180.0: {'right': {'Leaf': {'BBA': [0.34375, 0.125, 0.375, 0.15625, 0.0]}}, 'left': {'ConvexArea': {74.0: {'right': {'gaborSD_2_1': {54.6429: {'right': {'Sumaverage': {58.4908: {'right': {'Leaf': {'BBA': [0.0, 0.3269230769230769, 0.46153846153846156, 0.21153846153846154, 0.0]}}, 'left': {'Leaf': {'BBA': [0.0, 0.125, 0.6875, 0.125, 0.0625]}}}, 'BBA': [0.0, 0.25, 0.5476190476190477, 0.17857142857142858, 0.023809523809523808]}}, 'left': {'Leaf': {'BBA': [0.0625, 0.34375, 0.3125, 0.28125, 0.0]}}}, 'BBA': [0.017241379310344827, 0.27586206896551724, 0.4827586206896552, 0.20689655172413793, 0.017241379310344827]}}, 'left': {'Leaf': {'BBA': [0.0625, 0.5, 0.4375, 0.0, 0.0]}}}, 'BBA': [0.02702702702702703, 0.32432432432432434, 0.47297297297297297, 0.16216216216216217, 0.013513513513513514]}}}, 'BBA': [0.08333333333333333, 0.28888888888888886, 0.45555555555555555, 0.16111111111111112, 0.011111111111111112]}}, 'left': {'Leaf': {'BBA': [0.0, 0.21875, 0.1875, 0.4375, 0.15625]}}}, 'BBA': [0.07075471698113207, 0.2783018867924528, 0.41509433962264153, 0.2028301886792453, 0.0330188679245283]}}}, 'BBA': [0.056818181818181816, 0.23484848484848486, 0.375, 0.20075757575757575, 0.13257575757575757]}}}, 'BBA': [0.14, 0.21333333333333335, 0.35333333333333333, 0.17666666666666667, 0.11666666666666667]}}}, 'BBA': [0.14333333333333334, 0.19333333333333333, 0.35833333333333334, 0.17666666666666667, 0.12833333333333333]}}}, 'BBA': [0.14297658862876253, 0.1931438127090301, 0.3486622073578595, 0.18227424749163879, 0.13294314381270902]}}, 'left': {'SDIntensityBG': {268.4651: {'right': {'Eccentricity': {1.152305517: {'right': {'MinIntensityBG': {1942.0: {'right': {'Leaf': {'BBA': [0.8636363636363636, 0.06818181818181818, 0.0, 0.022727272727272728, 0.045454545454545456]}}, 'left': {'gaborSD_2_0': {42.1025: {'right': {'gaborSD_1_0': {90.8272: {'right': {'Leaf': {'BBA': [0.125, 0.40625, 0.46875, 0.0, 0.0]}}, 'left': {'Solidity': {9.737519565: {'right': {'gaborSD_3_1': {73.5: {'right': {'Leaf': {'BBA': [0.03125, 0.34375, 0.53125, 0.0625, 0.03125]}}, 'left': {'SecondMoment': {1.145374956: {'right': {'Contrast': {'BBA': [0.010869565217391304, 0.22826086956521738, 0.21739130434782608, 0.40217391304347827, 0.14130434782608695], 50.9835: {'right': {'Leaf': {'BBA': [0.016666666666666666, 0.23333333333333334, 0.26666666666666666, 0.4166666666666667, 0.06666666666666667]}}, 'left': {'Leaf': {'BBA': [0.0, 0.21875, 0.125, 0.375, 0.28125]}}}}}, 'left': {'Leaf': {'BBA': [0.0, 0.15625, 0.59375, 0.09375, 0.15625]}}}, 'BBA': [0.008064516129032258, 0.20967741935483872, 0.31451612903225806, 0.3225806451612903, 0.14516129032258066]}}}, 'BBA': [0.01282051282051282, 0.23717948717948717, 0.358974358974359, 0.2692307692307692, 0.12179487179487179]}}, 'left': {'Leaf': {'BBA': [0.0, 0.375, 0.5625, 0.0625, 0.0]}}}, 'BBA': [0.010638297872340425, 0.26063829787234044, 0.39361702127659576, 0.23404255319148937, 0.10106382978723404]}}}, 'BBA': [0.02727272727272727, 0.2818181818181818, 0.40454545454545454, 0.2, 0.08636363636363636]}}, 'left': {'Leaf': {'BBA': [0.0, 0.15625, 0.03125, 0.25, 0.5625]}}}, 'BBA': [0.023809523809523808, 0.26587301587301587, 0.35714285714285715, 0.20634920634920634, 0.14682539682539683]}}}, 'BBA': [0.14864864864864866, 0.23648648648648649, 0.30405405405405406, 0.17905405405405406, 0.13175675675675674]}}, 'left': {'MeanIntensityBG': {525.2666: {'right': {'Leaf': {'BBA': [0.9090909090909091, 0.022727272727272728, 0.045454545454545456, 0.022727272727272728, 0.0]}}, 'left': {'gaborSD_1_2': {'BBA': [0.05078125, 0.2265625, 0.37890625, 0.23046875, 0.11328125], 38.5497: {'right': {'gabormean_3_1': {44.5014: {'right': {'Perimeter': {197.0: {'right': {'Leaf': {'BBA': [0.0, 0.15625, 0.28125, 0.375, 0.1875]}}, 'left': {'gabormean_3_2': {95.1259: {'right': {'Leaf': {'BBA': [0.1875, 0.25, 0.53125, 0.03125, 0.0]}}, 'left': {'gaborSD_0_0': {292.0568: {'right': {'Leaf': {'BBA': [0.125, 0.28125, 0.28125, 0.3125, 0.0]}}, 'left': {'MaxIntensity': {'BBA': [0.020833333333333332, 0.3125, 0.4895833333333333, 0.17708333333333334, 0.0], 0.744047619: {'right': {'gaborSD_0_1': {331.65: {'right': {'Leaf': {'BBA': [0.0, 0.40625, 0.25, 0.34375, 0.0]}}, 'left': {'Leaf': {'BBA': [0.0, 0.21875, 0.65625, 0.125, 0.0]}}}, 'BBA': [0.0, 0.3125, 0.453125, 0.234375, 0.0]}}, 'left': {'Leaf': {'BBA': [0.0625, 0.3125, 0.5625, 0.0625, 0.0]}}}}}}, 'BBA': [0.046875, 0.3046875, 0.4375, 0.2109375, 0.0]}}}, 'BBA': [0.075, 0.29375, 0.45625, 0.175, 0.0]}}}, 'BBA': [0.0625, 0.2708333333333333, 0.4270833333333333, 0.20833333333333334, 0.03125]}}, 'left': {'Leaf': {'BBA': [0.03125, 0.0625, 0.34375, 0.3125, 0.25]}}}, 'BBA': [0.05803571428571429, 0.24107142857142858, 0.41517857142857145, 0.22321428571428573, 0.0625]}}, 'left': {'Leaf': {'BBA': [0.0, 0.125, 0.125, 0.28125, 0.46875]}}}}}}, 'BBA': [0.17666666666666667, 0.19666666666666666, 0.33, 0.2, 0.09666666666666666]}}}, 'BBA': [0.16275167785234898, 0.21644295302013422, 0.31711409395973156, 0.18959731543624161, 0.11409395973154363]}}, 'left': {'Compactness': {1.136143308: {'right': {'SDIntensity': {1602.0: {'right': {'Leaf': {'BBA': [0.96875, 0.0, 0.0, 0.03125, 0.0]}}, 'left': {'Area': {'BBA': [0.03676470588235294, 0.19117647058823528, 0.43014705882352944, 0.21691176470588236, 0.125], 367.0: {'right': {'Leaf': {'BBA': [0.0, 0.0625, 0.25, 0.1875, 0.5]}}, 'left': {'ConvexArea': {78.0: {'right': {'ConvexPerimeter': {216.0: {'right': {'Leaf': {'BBA': [0.03125, 0.03125, 0.5, 0.21875, 0.21875]}}, 'left': {'MeanIntensity': {0.8877547259999999: {'right': {'gaborSD_1_1': {326236.22: {'right': {'Leaf': {'BBA': [0.0625, 0.25, 0.5625, 0.125, 0.0]}}, 'left': {'gabormean_1_1': {4965.08: {'right': {'Leaf': {'BBA': [0.0, 0.3333333333333333, 0.36666666666666664, 0.3, 0.0]}}, 'left': {'Leaf': {'BBA': [0.0, 0.15625, 0.4375, 0.3125, 0.09375]}}}, 'BBA': [0.0, 0.2717391304347826, 0.391304347826087, 0.30434782608695654, 0.03260869565217391]}}}, 'BBA': [0.016129032258064516, 0.2661290322580645, 0.43548387096774194, 0.25806451612903225, 0.024193548387096774]}}, 'left': {'Leaf': {'BBA': [0.03125, 0.03125, 0.59375, 0.34375, 0.0]}}}, 'BBA': [0.019230769230769232, 0.21794871794871795, 0.46794871794871795, 0.27564102564102566, 0.019230769230769232]}}}, 'BBA': [0.02127659574468085, 0.18617021276595744, 0.4734042553191489, 0.26595744680851063, 0.05319148936170213]}}, 'left': {'Leaf': {'BBA': [0.16666666666666666, 0.3888888888888889, 0.4444444444444444, 0.0, 0.0]}}}, 'BBA': [0.044642857142857144, 0.21875, 0.46875, 0.22321428571428573, 0.044642857142857144]}}}}}}, 'BBA': [0.13486842105263158, 0.17105263157894737, 0.3848684210526316, 0.19736842105263158, 0.1118421052631579]}}, 'left': {'MaxIntensityBG': {537.9379: {'right': {'Leaf': {'BBA': [0.90625, 0.0, 0.03125, 0.0625, 0.0]}}, 'left': {'gaborSD_3_0': {67.2241: {'right': {'Leaf': {'BBA': [0.46875, 0.15625, 0.34375, 0.03125, 0.0]}}, 'left': {'Perimeter': {80.91168825: {'right': {'Leaf': {'BBA': [0.0, 0.09375, 0.21875, 0.21875, 0.46875]}}, 'left': {'MeanIntensityBG': {693.7437: {'right': {'Leaf': {'BBA': [0.15625, 0.09375, 0.28125, 0.34375, 0.125]}}, 'left': {'Area': {184.0: {'right': {'Leaf': {'BBA': [0.0, 0.21875, 0.28125, 0.375, 0.125]}}, 'left': {'gabormean_0_1': {5920.12: {'right': {'Leaf': {'BBA': [0.03125, 0.375, 0.5, 0.0, 0.09375]}}, 'left': {'Energy': {67.4516: {'right': {'gabormean_1_0': {4451.82: {'right': {'Leaf': {'BBA': [0.0625, 0.3125, 0.4375, 0.1875, 0.0]}}, 'left': {'Leaf': {'BBA': [0.0, 0.09375, 0.65625, 0.21875, 0.03125]}}}, 'BBA': [0.0375, 0.225, 0.525, 0.2, 0.0125]}}, 'left': {'Leaf': {'BBA': [0.03125, 0.25, 0.71875, 0.0, 0.0]}}}, 'BBA': [0.03571428571428571, 0.23214285714285715, 0.5803571428571429, 0.14285714285714285, 0.008928571428571428]}}}, 'BBA': [0.034722222222222224, 0.2638888888888889, 0.5625, 0.1111111111111111, 0.027777777777777776]}}}, 'BBA': [0.028409090909090908, 0.2556818181818182, 0.5113636363636364, 0.1590909090909091, 0.045454545454545456]}}}, 'BBA': [0.04807692307692308, 0.23076923076923078, 0.47596153846153844, 0.1875, 0.057692307692307696]}}}, 'BBA': [0.041666666666666664, 0.2125, 0.44166666666666665, 0.19166666666666668, 0.1125]}}}, 'BBA': [0.09191176470588236, 0.20588235294117646, 0.43014705882352944, 0.17279411764705882, 0.09926470588235294]}}}, 'BBA': [0.17763157894736842, 0.18421052631578946, 0.3881578947368421, 0.1611842105263158, 0.08881578947368421]}}}, 'BBA': [0.15625, 0.17763157894736842, 0.38651315789473684, 0.17927631578947367, 0.10032894736842106]}}}, 'BBA': [0.15946843853820597, 0.19684385382059802, 0.3521594684385382, 0.18438538205980065, 0.10714285714285714]}}}, 'BBA': [0.15125, 0.195, 0.35041666666666665, 0.18333333333333332, 0.12]}}
    print("BDT TREE: ",file = f)
    print(tree, file = f)
    
    # Graph Tree
    #graph = pydot.Dot(graph_type='graph')
    #visit(tree)
    #graph.write_png("output/BDT.png")
    
    # Get actual data
    actualTrain = getPigns(train_features)
    actualTest = getPigns(test_features)
    
    # Classify training set
    print ("Classifying Training Set...") 
    for i in range(0,len(train_features)):
            trainLabels.append(classify(tree, test_header,train_features[i]))
    
    # Classify calibration set & compute conformity
    print ("Classifying Calibration Set...") 
    for i in range(0,len(calib_features)):
            calibLabels.append(classify(tree, test_header,calib_features[i]))
    
    # Calibration Conformity
    print("Computing Calibration Conformity...")
    actualCalib = getPigns(calib_features)
    calib_conf = Train_Conformity(actualCalib, calibLabels)
    print("\nCALIBRATION:", file=f)
    for i in range (0,len(actualCalib)):
        print("Probabilities: ", actualCalib[i], "\tConformity Score = ", calib_conf[i], file=f)
    
    # Classify testing set
    print ("Classifying Testing Set...") 
    for i in range(0,len(test_features)):
            testLabels.append(classify(tree, test_header,test_features[i]))
            
    # Compute final conformity and p-values
    print("Computing Testing Conformity...")
    test_conf = Test_Conformity(testLabels)
    p_vals = PValue(test_conf,calib_conf)
    print("\nTESTING CONFORMITY:", file=f)
    confidence = []
    credibility = []
    for i, val in enumerate(p_vals):
        m = val.index(max(val))
        sec = max([x for k,x in enumerate(val) if k != m])
        #mystr = str(i) + "  P-values: " + '{1:4f}'.format(val) + "  Class Label: " + str(getMax(val)) + "  Confidence: " + '{1:4f}'.format(str(1-sec)) + "\tCredibility: " + '{1:4f}'.format(max(val))
        print(i, "\tP-values: [", '{d[0]:.5} {d[1]:.5} {d[2]:.5} {d[3]:.5} {d[4]:.5}'.format(d=val), "]\tClass Label: ", val.index(max(val))+1, "  Confidence: ", 1-sec, "\tCredibility: ", max(val), file=f)
        
        confidence.append(1 - sec)
        credibility.append(max(val))
        
    # Output data to csv and text files
    print ("Writing Data...") 
    trainhead = "Training Data: (d = " + str(maxdepth) + " | np = " + str(nparent) + " | nc = " + str(nchild) + ")"
    testhead = "Testing Data: (d = " + str(maxdepth) + " | np = " + str(nparent) + " | nc = " + str(nchild) + ")"
    writeData("Training", trainhead, "output/TrainOutput.csv", "wb", actualTrain, trainLabels, confidence, credibility, 0)
    writeData("Testing", testhead, "output/TestOutput.csv", "wb", actualTest, testLabels, confidence, credibility, len(trainLabels))
    
    #increase round of k-fold vlaidation
    k_round += 1
# Close output file
f.close()
print("DONE")