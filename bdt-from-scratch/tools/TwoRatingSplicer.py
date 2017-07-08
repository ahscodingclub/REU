# -*- coding: utf-8 -*-
"""
Created on Thu Aug 04 13:54:27 2016

@author: EBARNS
"""
import csv
import pandas as pd
from imblearn.over_sampling import ADASYN

def getPigns(dataset):
    nodePigns = []
    # For each case at this node, calculate a vector of probabilities
    for k, case in enumerate(dataset):
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
        pign = five_to_two(pign)
        nodePigns.append(pign) # Add pign to list of pigns
    return nodePigns

def five_to_two(lst):
    theta = lst[2]/2
    pb = lst[0] + (lst[1] * 0.75) + theta + (lst[3] * 0.25)
    pm = lst[4] + (lst[3] * 0.75) + theta + (lst[1] * 0.25)
    return [pb,pm]
    
def main():    
    set_name = "./data/LIDC_FULLSET.csv"
    id_data  = pd.read_csv(set_name, header =0)
    set_header = list(id_data.columns.values)
    id_data = id_data._get_numeric_data()
    set_data_array = id_data.as_matrix()
    set_data_array = set_data_array.tolist()
    set_data_array = [case for case in set_data_array if case.count(0) <= 5 and case[-4:].count(0) == 1]
    balance = [[],[]]
    with open("./output/FULL_EDIT_LIDC.csv","wb") as csvfile:
        i = 0
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(set_header)
        data = getPigns(set_data_array)
        for i in range(len(data)):
            classification = data[i].index(max(data[i]))
            balance[classification].append(set_data_array[i])
            
           
        distribution = [len(lst) for lst in balance]
        print distribution
        balance_thresh = min(distribution)
        new_set = []
        for j in range(len(balance)):
            for i in range(balance_thresh):
                new_set.append(balance[j][i]) 

        for i in range(3):
            for row in new_set:
                writer.writerow(row)
        #            writer.writerow(row + [m])
        print len(new_set) * 3
        
main()
        