# -*- coding: utf-8 -*-
"""
Created on Thu Aug 04 13:54:27 2016

@author: EBARNS
"""
import csv
import pandas as pd
import numpy as np
from collections import Counter
set_name = "./data/LIDC_FULLSET.csv"
id_data  = pd.read_csv(set_name, header =0)
set_header = list(id_data.columns.values)
id_data = id_data._get_numeric_data()
set_data_array = id_data.as_matrix()
set_data_array = set_data_array.tolist()

balance = [[],[],[],[],[],[]]
with open("./output/FULL_EDIT_LIDC.csv","wb") as csvfile:
    i = 0
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(set_header)
    for row in set_data_array:
        rw = [k for k in row[-4:] if k != 0]
        rw = Counter(rw)
        mode = rw.most_common(1)
        if mode != []: 
            m = int(mode[0][0])
        else: m = int(6) # appends to an error list 
        if(row.count(0) <= 5 and row[-4:].count(0) >= 0):
            i += 1
            balance[m-1].append(row)
    distribution = [len(lst) for lst in balance]
    print distribution
    balance_thresh = min(distribution[:-1])
    new_set = []
    for j in range(len(balance)-1):
        for i in range(balance_thresh):
            new_set.append(balance[j][i]) 
    for row in new_set:
        writer.writerow(row)
#            writer.writerow(row + [m])
    print len(new_set)