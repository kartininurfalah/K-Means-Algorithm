#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 09:38:57 2018

@author:kartini
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def newList(w, h):
    return [[0 for x in range(w)] for y in range(h)]

trainResult = np.array(pd.read_csv('TestsetTugas2.txt', header=None).values)

trainResult = list(trainResult)

separatedResult = newList(0, 13)
separatedSSE = newList(0, 13)

for i, item in enumerate(trainResult): 
    item = list(item)
    trainResult[i] = item
    separatedResult[item[0]].append(item[2])

for x, sse in enumerate(separatedResult):
    if(sse):    
        separatedSSE[x] = min(sse)

del separatedSSE[0]

print(separatedSSE)

plt.plot(np.arange(1,13), separatedSSE, 's-')
plt.xlabel('K')
plt.ylabel('SSE')
plt.show()

#sse = list(trainResult[:,2]

#resultSSE = sse.index(min(sse))

#minimumSSECentroid = trainResult[resultSSE][0]

#for i, trainItem in enumerate(trainResult):


zArray = list(trainResult)

#print(sse)