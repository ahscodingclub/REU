def cross_product(X,Y):
    product = [ [ 0 for y in range(len(Y)) ] for x in range(len(X)) ]
    # iterate through rows of X
    for i in range(len(X)):
        # iterate through rows of Y
        for j in range(len(Y)):
            product[i][j] += X[i] * Y[j]
    return product

def getAccuracy(class_matrix): # returns accuracy of misclassification matrix 
    accy = 0.0
    for j in range(0,5):
        accy += class_matrix[j][j]
    return accy / sum2D(class_matrix)

def sum2D(input):
    return sum(map(sum, input))

x = [0]*5

for i in range(len(x)):
    x[i] = input("x: ")

y=x

cross = cross_product(x,y)

for row in cross:
    print(["{0:5.5}".format(str(val)) for val in row])

print("Max Accuracy = ", '{:.4f}'.format(getAccuracy(cross)* 100, "%"))
