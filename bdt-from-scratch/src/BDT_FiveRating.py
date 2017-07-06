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

# calculate a confusion matrix given predicted and actual, using full complexity of BBA
# returns an array of confusion matrices
def getConfusionMatrix(predicted,actual):
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

# Calculate accuracy given a confusion matrix
def getAccuracy(class_matrix): # returns accuracy of misclassification matrix 
    accy = 0.0
    for j in range(0,5):
        accy += class_matrix[j][j]
    return accy / sum2D(class_matrix)

# Sum elements in a 2D array
def sum2D(input):
    return sum(map(sum, input))

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
def writeData(trainOrTest, header, filename, params, actual, predicted, conf, cred, classes, accuracy, p_vals, id_start):
    with open(filename, params) as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if trainOrTest == "Training":
            writer.writerow(['Nodule ID',\
                             'Actual [1]',      'Actual [2]',       'Actual [3]',       'Actual [4]',       'Actual [5]',\
                             'Predicted [1]' ,  'Predicted [2]',    'Predicted [3]',    'Predicted [4]',    'Predicted [5]',\
                              header])
        else:
            writer.writerow(['Nodule ID',\
                             'Actual [1]',      'Actual [2]',       'Actual [3]',       'Actual [4]',       'Actual [5]',\
                             'Predicted [1]' ,  'Predicted [2]',    'Predicted [3]',    'Predicted [4]',    'Predicted [5]',\
                             'Confidence', 'Credibility', 'Sureness of Case', 'Classes', header])
            testConform = Train_Conformity(predicted, actual)

        
        for i in range(0, len(predicted)):
            # Write data and similarity measures to file
            if trainOrTest == "Training":
                writer.writerow([set_data_array[i+id_start][2],\
                                 actual[i][0], actual[i][1], actual[i][2], actual[i][3], actual[i][4],\
                                 predicted[i][0], predicted[i][1], predicted[i][2], predicted[i][3], predicted[i][4],\
                                ])
            else:
                writer.writerow([set_data_array[i+id_start][2],\
                                 actual[i][0], actual[i][1], actual[i][2], actual[i][3], actual[i][4],\
                                 predicted[i][0], predicted[i][1], predicted[i][2], predicted[i][3], predicted[i][4],\
                                 conf[i], cred[i], testConform[i], classes[i]])
        
        # For each case at this node
        #print(len(actual),len(predicted))
     
    # Calculate Accuracy and AUCdt



    confusion = getConfusionMatrix(predicted, actual)
    myAccuracy = getAccuracy(confusion)
    
    # Output to Console
    print("\n" + trainOrTest + "\n")
    for row in confusion:
        print(["{0:5.5}".format(str(val)) for val in row])
    print(trainOrTest,"Accuracy = ", '{:.4f}'.format(float(myAccuracy * 100)), "%")
    
    # Output Confusion Matrices, Accuracies, AUCdt, and ROC AUC
    print("\n", trainOrTest, "Confusion Matrix", file=f)
    for row in confusion:
        print(["{0:5.5}".format(str(val)) for val in row], file=f)
    print("Accuracy = ", '{:.4f}'.format(float(myAccuracy * 100)), "%", file=f)

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
def plotVio(old, category, a, axes, xlabel, ylabel):
    new = []
    labels = []
    
    # Get labels from category
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
    
    axes[a].yaxis.grid(True)
    axes[a].set_xticks([y+1 for y in range(0,len(new))])
    axes[a].set_xticklabels(labels)    
    axes[a].set_xlabel(xlabel)
    axes[a].set_ylabel(ylabel)
    axes[a].set_ylim([0,102])
    #return v

# DRAW VIOLIN PLOTS FOR CONFIDENCE / CREDIBILITY
def violin(conf, cred, category, xlabel, show):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    # Confidence
    plotVio(conf, category, 0, axes,  xlabel, 'Confidence')
    axes[0].set_title('Confidence Values at Each ' + xlabel)
    
    # Credibility
    plotVio(cred, category, 1, axes,  xlabel, 'Credibility')
    axes[1].set_title('Credibilty Values at Each ' + xlabel)
    
    # Show & save figure
    figure_dest = f_name[0:-4]+'.png'
    plt.savefig(figure_dest, bbox_inches='tight')
    print("Figure saved in ", figure_dest)
    if show == 'show' and show != 'hide':
        plt.show()
    
#####################################
# MAIN SCRIPT: Build, Classify, Output
#####################################       
# Setup
#input loop for dist settings
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
k_best = [0,0.0,[],[],[],[],[],[]]
trainhead = "Training Data: (d = " + str(maxdepth) + " | np = " + str(nparent) + " | nc = " + str(nchild) + " | k = " + str(kfolds) + ")"
testhead = "Testing Data: (d = " + str(maxdepth) + " | np = " + str(nparent) + " | nc = " + str(nchild) + " | k = " + str(kfolds) + ")"
print("Classifying with BDT Parameters (d = ",maxdepth,", np = ",nparent,", nc = ",nchild,", k = ",kfolds,"):\n",file=f)

global graph
for trn_ind, tst_ind in kf:
    trainLabels = []
    testLabels = []
    calibLabels = []
    setTrain(LIDC_Data[trn_ind])
    test_features = LIDC_Data[tst_ind].tolist()
    
    # Console Output
    print("\n K-FOLD VALIDATION ROUND ",k_round," OF ",kfolds)
    print("#################################")
    print("Train Size: ", len(train_features))
    print("Calibration Size: ", len(calib_features))
    print("Test Size: ", len(test_features))
    print ("Building Belief Decision Tree. ..") 
    
    # Create Tree
    # setting "switch = True" will make new tree each time
    tree = getTrees(train_features, header, nparent, nchild, 0, maxdepth, True) 

    #graphing the tree
#    graph = pydot.Dot(graph_type='graph')
#    visit(tree)
#    graph.write_png("BDT.png")
    
    # Get actual data
    actualTrain = getPigns(train_features)
    actualTest = getPigns(test_features)
    
    # Classify training set
    print ("Classifying Training Set...") 
    for i in range(0,len(train_features)):
            trainLabels.append(classify(tree, test_header,train_features[i]))
    
    # Classify calibration set
    print ("Classifying Calibration Set...") 
    for i in range(0,len(calib_features)):
            calibLabels.append(classify(tree, test_header,calib_features[i]))
    
    # Compute Calibration Conformity
    print("Computing Calibration Conformity...")
    actualCalib = getPigns(calib_features)
    calib_conf = Train_Conformity(actualCalib, calibLabels)
    
    # Classify testing set
    print ("Classifying Testing Set...") 
    for i in range(0,len(test_features)):
            testLabels.append(classify(tree, test_header,test_features[i]))
            
    # Compute final conformity and p-values
    print("Computing Testing Conformity...")
    test_conf = Test_Conformity(testLabels)
    p_vals = PValue(test_conf,calib_conf)

    confidence = []
    credibility = []
    classes = []
    
    for i, val in enumerate(p_vals):
        m = val.index(max(val))
        sec = max([x for k,x in enumerate(val) if k != m])
        classes.append(testLabels[i].index(max(testLabels[i]))+1)        
        confidence.append(1 - sec)
        credibility.append(max(val))
    
    # increase round of k-fold vlaidation
    conf_matrix = getConfusionMatrix(testLabels, actualTest)
    accy = getAccuracy(conf_matrix)
    if accy > k_best[1]:
        k_best = [k_round, accy, actualTrain, trainLabels, actualTest, testLabels, confidence, credibility, classes]
    k_round += 1

# Output data to csv and text files
accuracy = k_best[1]
actualTrain = k_best[2]
trainLabels = k_best[3]
actualTest = k_best[4]
testLabels = k_best[5]
confidence = k_best[6]
credibility = k_best[7]
classes = k_best[8]

print("length of actual test: ", len(actualTest))
print("length of train_labels: ", len(trainLabels)) 
print("length of p vals: ", len(p_vals)) 

print ("\nWriting Data for best fold k =", k_best[0], "...") 

writeData("Training", trainhead, f_name[0:-4] + '_training' + '.csv', "wb", actualTrain, trainLabels, confidence, credibility, classes, accuracy, p_vals, 0) 

writeData("Testing", testhead, f_name[0:-4] + '_testing' + '.csv', "wb", actualTest, testLabels, confidence, credibility, classes, accuracy, p_vals, len(trainLabels))


violin([100*x for x in confidence], [100*x for x in credibility], classes, 'Classification', 'hide')
    
# Close output file
f.close()
print("DONE")
