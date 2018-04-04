import nltk
import numpy as np
import pandas as pd
import math
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

dTrainX1 = dataTrain[:,0]
dTrainX2 = dataTrain[:,1]
dtTrain = [dTrainX1, dTrainX2]

dTestX1 = dataTest[:,0]
dTestX2 = dataTest[:,1]
dtTest = [dTestX1, dTestX2]
# =============================================================================
# # datatrain and dataTest scater
# =============================================================================
X = np.array(list(zip(dTrainX1, dTrainX2)))

plt.scatter(dTestX1, dTestX2)
plt.show()

k = 5
indexCentroid= randint(1,688)
centroid = X[indexCentroid]
plt.scatter(dTrainX1, dTrainX2)
plt.scatter(centroid[0], centroid[1], marker = '*', s = 200)
# =============================================================================
# Euclidian distance function
# =============================================================================
def euclidFunct(x, y, ax=1):
    return np.linalg.norm(x - y, axis = ax)

def kMeansAlgo():
    dtCluster = []
    
