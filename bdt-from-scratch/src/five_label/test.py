from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

clf=DecisionTreeClassifier()
train = [[1,1,2,3,4],[2,2,3,4,5]]
train_lab = [1,2]
test = [[1,2,3,4,5],[2,2,3,4,5]]
test_lab = [2,2]
clf.fit(train,train_lab)
output=clf.predict(test)

print accuracy_score(test_lab,output)


