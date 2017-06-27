import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

rank_1 = 0
rank_2 = 0
rank_3 = 0
rank_4 = 0
rank_5 = 0

f = open("ModeBalanced_170_LIDC_809_Random.csv")
df = pd.read_csv(f)
malig_1 = df.Malignancy_1
malig_2 = df.Malignancy_2
malig_3 = df.Malignancy_3
malig_4 = df.Malignancy_4

for i in range(len(malig_1)):
    if malig_1[i] == 1 or malig_2[i] == 1 or malig_3[i] == 1 or malig_4[i] == 1:
        rank_1+=1
    if malig_1[i] == 2 or malig_2[i] == 2 or malig_3[i] == 2 or malig_4[i] == 2:
        rank_2+=1
    if malig_1[i] == 3 or malig_2[i] == 3 or malig_3[i] == 3 or malig_4[i] == 3:
        rank_3+=1
    if malig_1[i] == 4 or malig_2[i] == 4 or malig_3[i] == 4 or malig_4[i] == 4:
        rank_4+=1
    if malig_1[i] == 5 or malig_2[i] == 5 or malig_3[i] == 5 or malig_4[i] == 5:
        rank_5+=1

print("ratings of rank 1: ", rank_1)
print("ratings of rank 2: ", rank_2)
print("ratings of rank 3: ", rank_3)
print("ratings of rank 4: ", rank_4)
print("ratings of rank 5: ", rank_5)

objects = ('1', '2', '3', '4', '5')
y_pos = np.arange(len(objects))
performance = [rank_1,rank_2,rank_3,rank_4,rank_5]
  
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.xlabel('malignancy ratings')
plt.title('Malignancy Rating Distribution')
   
plt.show()
