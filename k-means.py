import nltk
import numpy as np
import pandas as pd
import math
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import genfromtxt
from sklearn.model_selection import train_test_split
from random import randint


# =============================================================================
# #load data train
# =============================================================================
dataTrain = np.genfromtxt("/home/helmisatria/FALAH/Machine Learning/K-means/TrainsetTugas2.txt")
dataTest = np.genfromtxt("/home/helmisatria/FALAH/Machine Learning/K-means/TestsetTugas2.txt")

dTrainX = dataTrain[:,0]
dTrainY = dataTrain[:,1]
dtTrain = [dTrainX, dTrainY]

dTestX = dataTest[:,0]
dTestY = dataTest[:,1]
dtTest = [dTestX, dTestY]
# =============================================================================
# # datatrain and dataTest scater
# =============================================================================
X = np.array(list(zip(dTrainX, dTrainY)))
#
plt.scatter(dTestX, dTestY)
plt.show()

k = 5
#
def initialCentroid(k):    
    centroidTemp = []
    for i in range(k):
       indexCentroid= randint(1,688)
       centroid = dataTrain[indexCentroid]
       centroidTemp.append(centroid)
    return centroidTemp

def newList(w,h):
    return [[0 for x in range(w)] for y in range(h)]

tempC = initialCentroid(k)
centroidX = [item[0] for item in tempC]
centroidY = [item[1] for item in tempC]

#plt.scatter(dTrainX, dTrainY)
#plt.scatter(centroidX, centroidY, marker = '*', s = 200)
print(tempC)
# =============================================================================
# Euclidian distance function
# =============================================================================
def euclidFunct(xa, xb, ya, yb):
    return sqrt((xa-xb)**2 + (ya-yb)**2)

#def kMeansAlgo(dTrain, centroid):
#    temp = [0,0]*k
#    while (temp != centroid):
#        dtCluster = []
##        temp = centroid
#        for i,point in enumerate(dtTrain):
#            tempDist = []
#            for j,pCentroid in enumerate(centroid):
#                tempDist.append(euclidFunct(point[0], pCentroid[0], point[1], pCentroid[1]))
#            minDist = min(tempDist)
#            indexCentroid = tempDist.index(minDist)
#            dtCluster.append(indexCentroid)
#            print(tempDist) 
#    return dtCluster
newCluster = [[0,0]] * k
loop = 0
while(np.any(np.not_equal(tempC, newCluster)) and (loop!=70)):
    dtCluster = [] 
    if(newCluster[0] != [0,0]):
        tempC = newCluster                      
    for i, point in enumerate(dataTrain):
        tempDist = []
        for j,pCentroid in enumerate(tempC):
            tempDist.append(euclidFunct(point[0], pCentroid[0], point[1], pCentroid[1]))
        minDist = min(tempDist)
        indexCentroid = tempDist.index(minDist)
        dtCluster.append(indexCentroid)
    #    print(dtCluster)
        
    #clust = kMeansAlgo(dtTrain, tempC)
    #print(clust)
    
    mergeDataClust = np.column_stack((dataTrain, dtCluster))
    separatedCluster = newList(0, k)
    
    for p,xPoint in enumerate(mergeDataClust):
        separatedCluster[int(xPoint[2])].append(xPoint)
    
    
    for x,pClust in enumerate(separatedCluster):
        dataX = []
        dataY = []
        for i,valClust in enumerate(pClust):
            dataX.append(valClust[0])    
            dataY.append(valClust[1])
        newCluster[x] = [np.mean(dataX), np.mean(dataY)]
    loop = loop+1
    
print(newCluster)

clusterX = [item[0] for item in newCluster]
clusterY = [item[1] for item in newCluster]

plt.scatter(dTrainX, dTrainY, c=dtCluster)
plt.scatter(clusterX, clusterY, marker = '*', s = 333)
plt.show()

def sse():
    sum = 0
    for i,valCluster in enumerate(separatedCluster):
        for j, pClust in enumerate(valCluster):
            sum += euclidFunct(tempC[i][0], pClust[0], tempC[i][1], pClust[1])
    return sum


print('SSE : ',sse())

sseCluster = [k, tempC, sse()]

sseCluster = pd.DataFrame([sseCluster])
finalResult = pd.DataFrame(separatedCluster)
finalResult.to_csv('finalResult.csv', header=False, index=False)

    