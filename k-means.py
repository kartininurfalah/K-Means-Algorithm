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
#plt.scatter(dTestX, dTestY)
#plt.show()

k = 2
#
def initialCentroid(k):    
    centroidTemp = []
    for i in range(k):
       indexCentroid= randint(1,688)
       centroid = dataTrain[indexCentroid]
       centroidTemp.append(centroid)
    return centroidTemp

tempC = initialCentroid(k)
centroidX = [item[0] for item in tempC]
centroidY = [item[1] for item in tempC]

plt.scatter(dTrainX, dTrainY)
plt.scatter(centroidX, centroidY, marker = '*', s = 200)
print(tempC)
# =============================================================================
# Euclidian distance function
# =============================================================================
def euclidFunct(xa, xb, ya, yb):
    return sqrt((xa-xb)**2 + (ya-yb)**2)


def kMeansAlgo(dTrain, centroid):
    temp = [0,0]*k
    while (temp != centroid):
        dtCluster = []
        temp = centroid
        for i,point in enumerate(dtTrain):
            tempDist =[]
            for j,pCentroid in enumerate(centroid):
                tempDist.append(euclidFunct(point[0], pCentroid[0], point[1], pCentroid[1]))
            minDist = min(tempDist)
            indexCentroid = tempDist.index(minDist)
            dtCluster.append(indexCentroid)
            newCentroid = 0
    return dtCluster

clust = kMeansAlgo(dtTrain, tempC)
print(clust)