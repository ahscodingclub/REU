# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 13:47:10 2016

@author: EBARNS
"""

########
#REMOVE THE TAG ON THE RIGHT SIDE OF THE DATA SET IF IT'S THERE 
#OTHERWISE ACCURACY WILL BE OFF AND COSINE WON'T COMPUTE 
########
import pandas as pd
import csv
from sklearn.metrics import confusion_matrix # Assess misclassification
from scipy import spatial
def importIdData(set_name):
    id_data  = pd.read_csv(set_name, header =0)
    set_header = list(id_data.columns.values)
    print(set_header)
    id_data = id_data._get_numeric_data()
    set_data_array = id_data.as_matrix()
    return set_data_array
def find_outliers(dataset, threshold):
    clean_set = []
    outlier_set = []
    for case in dataset:
        print(case[-2])
        if(case[-2] > threshold):
            outlier_set.append(case)
        else:
            clean_set.append(case)
    
    return clean_set, outlier_set


# Calculate a confusion matrix given predicted and actual
def getConfusionMatrix(predicted, actual): #returns misclassification matrix
    # TEST: Find greatest probability to use as prediction
    pred_proba = [(case.index(max(case))+1) for case in predicted]
    act_proba = [(case.index(max(case))+1) for case in actual]
    return confusion_matrix(act_proba,pred_proba)
    
    
def CosineSimilarity(actual, predicted):
    similarities= []
    average = 0.0
    for i in range(len(predicted)):
        temp = spatial.distance.cosine(actual[i],predicted[i])
        similarities.append(temp)
        average += temp
        
    return average/len(predicted), similarities
# Calculate accuracy given a confusion matrix
def getAccuracy(class_matrix, num_cases): # returns accuracy of misclassification matrix 
    accy = 0.0
    for j in range(0,5):
        accy += class_matrix[j][j]
    return accy / num_cases
            
def writeData(trainOrTest,filename, params, dataset):
    with open(filename, params) as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['NoduleID','Actual1', 'Actual2', 'Actual3', 'Actual4', 'Actual5', 'Predicted1' , 'Predicted2', 'Predicted3', 'Predicted4', 'Predicted5', 'RMSE', 'PearsonR'])
        for row in dataset:
            writer.writerow(row)
data_trn = importIdData("./output/TrainOutput.csv")
data_tst = importIdData("./output/TestOutput.csv")
clean_trn, out_trn = find_outliers(data_trn,0.8)
clean_tst, out_tst = find_outliers(data_tst,0.8)
writeData("Training ","output/TrainOutput_NoOutlier.csv", "wb", clean_trn)
writeData("Training ","output/TrainOutput_Outlier.csv", "wb", out_trn)
writeData("Testing ","output/TestOutput_NoOutlier.csv", "wb", clean_tst)
writeData("Testing ","output/TestOutput_Outlier.csv", "wb", out_tst)

no_outliers = importIdData('./output/TrainOutput_NoOutlier.csv')
actual = [case[1:6].tolist() for case in no_outliers]
predicted = [case[6:-2].tolist() for case in no_outliers]


conf_matrix = getConfusionMatrix(predicted, actual)
accy = getAccuracy(conf_matrix, len(predicted))
cos_sim, sims = CosineSimilarity(actual, predicted)
print("Training Output W/ Outliers Removed")
print(conf_matrix)
print("Accuracy: ", accy)
print "Average Cosine Similarity: ", cos_sim

no_outliers = importIdData('./output/TestOutput_NoOutlier.csv')
actual = [case[1:6].tolist() for case in no_outliers]
predicted = [case[6:-2].tolist() for case in no_outliers]


conf_matrix = getConfusionMatrix(predicted, actual)
accy = getAccuracy(conf_matrix, len(predicted))
cos_sim, sims = CosineSimilarity(actual, predicted)
print("Testing Output W/ Outliers Removed")
print(conf_matrix)
print("Accuracy: ", accy)
print "Average Cosine Similarity: ", cos_sim
print len([x for x in sims if x == 1])

