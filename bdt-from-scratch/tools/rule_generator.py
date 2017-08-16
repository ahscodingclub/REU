import pickle
import pandas as pd
import csv

#####################################
# Classifying NEW CASES
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
            decision.append(firstStr + " > " + str(key))
            return classify(secondDict[key][k],featLabels,testVec) # GO RIGHT
        else: # GET SUBTREE
            k = secondDict[key].keys()[1]
            decision.append(firstStr + " < " + str(key))
            return classify(secondDict[key][k],featLabels,testVec) # GO LEFT

feature_labels = ['noduleID','Area', 'ConvexArea', 'Perimeter', 'ConvexPerimeter', 'EquivDiameter', 'MajorAxisLength', 'MinorAxisLength', 'Elongation', 'Compactness', 'Eccentricity', 'Solidity', 'Extent', 'Circularity', 'RadialDistanceSD', 'SecondMoment', 'Roughness', 'MinIntensity', 'MaxIntensity', 'MeanIntensity', 'SDIntensity', 'MinIntensityBG', 'MaxIntensityBG', 'MeanIntensityBG', 'SDIntensityBG', 'IntensityDifference', 'markov1', 'markov2', 'markov3', 'markov4', 'markov5', 'gabormean_0_0', 'gaborSD_0_0', 'gabormean_0_1', 'gaborSD_0_1', 'gabormean_0_2', 'gaborSD_0_2', 'gabormean_1_0', 'gaborSD_1_0', 'gabormean_1_1', 'gaborSD_1_1', 'gabormean_1_2', 'gaborSD_1_2', 'gabormean_2_0', 'gaborSD_2_0', 'gabormean_2_1', 'gaborSD_2_1', 'gabormean_2_2', 'gaborSD_2_2', 'gabormean_3_0', 'gaborSD_3_0', 'gabormean_3_1', 'gaborSD_3_1', 'gabormean_3_2', 'gaborSD_3_2', 'Contrast', 'Correlation', 'Energy', 'Homogeneity', 'Entropy', 'x_3rdordermoment', 'Inversevariance', 'Sumaverage', 'Variance', 'Clustertendency', 'MaxProbability']

nodule_data = pd.read_csv("/home/x1/Documents/REU/bdt-from-scratch/data/modeBalanced/ModeBalanced_170_LIDC_809_Random.csv")
nodule_data = pd.DataFrame.as_matrix(nodule_data)

tree_files = ['/home/x1/server_data/tree_1_1.pkl',
              '/home/x1/server_data/tree_1_2.pkl',
              '/home/x1/server_data/tree_1_3.pkl',
              '/home/x1/server_data/tree_1_4.pkl',
              '/home/x1/server_data/tree_1_5.pkl',
              '/home/x1/server_data/tree_1_6.pkl',
             ]

testing_files = ['/home/x1/server_data/test_1_1_testing.csv',
                 '/home/x1/server_data/test_1_2_testing.csv',
                 '/home/x1/server_data/test_1_3_testing.csv',
                 '/home/x1/server_data/test_1_4_testing.csv',
                 '/home/x1/server_data/test_1_5_testing.csv',
                 '/home/x1/server_data/test_1_6_testing.csv',
                ]


decisions = []
for i in range(0, len(tree_files)):
    tree_file = tree_files[i]
    testing_file = testing_files[i]
    """
    load the tree from file
    """
    with open(tree_file, 'rb') as f:
        tree = pickle.load(f)
    """
    get the nodule IDs from the testing file
    assoicated with this specific tree
    """
    nodule_IDs = pd.read_csv(testing_file)['Nodule ID']
    """
    get the associated features with the nodule IDs from"
    the data frame
    """
    cases = [nodule_data[0:len(nodule_IDs)]][0]
    nodule_data = nodule_data[len(nodule_IDs):]
    """
    run these nodule ids through their associated tree
    to get the decision made for these cases
    """
    for case in cases:
        decision = []
        classify(tree, feature_labels, case[:-4])
        decisions.append(decision)

f = open("/home/x1/Documents/REU/bdt-from-scratch/data/modeBalanced/ModeBalanced_170_LIDC_809_Random (copy).csv", "r")
csv_in = csv.reader(f)
header_row = next(csv_in)
rows = []
for row in csv_in:
    rows.append(row)

f.close()

f = open("/home/x1/Documents/REU/bdt-from-scratch/data/modeBalanced/ModeBalanced_170_LIDC_809_Random (copy)_edit.csv", "w")
csv_out = csv.writer(f, delimiter=",")
csv_out.writerow(header_row)

for i in range(0, len(rows)):
    rows[i].append(decisions[i])
    csv_out.writerow(rows[i])

f.close()
