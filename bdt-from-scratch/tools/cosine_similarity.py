from scipy import spatial

dataSetI = [3, 45, 7, 2]
dataSetII = [0, 0, 1, 0]
result = 1 - spatial.distance.cosine(dataSetI, dataSetII)

print("result: ", result)
