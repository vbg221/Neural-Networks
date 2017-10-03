# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 16:33:29 2017

Name : Vatsal Gopani
N ID : N17368916
NetID: vbg221

Description : Assignment 3.1 in Neural Network.
              Testing the use of Perceptron for 
              simple classificatin. 

@author: vbg22
"""

import numpy as np
import pandas as pd

dt = pd.read_csv("C:/Users/vbg22/Downloads/Pictures/PerceptronDataF17.txt", names = ["X1", "X2", "X3", "X4", "Y"], delim_whitespace = True)

maxContinuousCount = 0
loopIterator = 0

#Initial Weights:
w = [0,0,0,0,0]     #w[0] is bias
theta = 0.05
x = [0,0,0,0,0]
alpha = 1
numberOfIterations = 0

"""
Training the Neurons so that weight can be adjusted and be used later on for prediction
"""
while(maxContinuousCount != 1000):
    #Taking input from the DataFrame.    
    x[0] = 1
    x[1] = dt.ix[loopIterator]["X1"]
    x[2] = dt.ix[loopIterator]["X2"]
    x[3] = dt.ix[loopIterator]["X3"]
    x[4] = dt.ix[loopIterator]["X4"]
    
    y = dt.ix[loopIterator]["Y"]
    
    yin = np.dot(w, x)
    
    if(yin >= theta):
        yin = 1
    elif(yin <= -theta):
        yin = -1
    else:
        yin = 0
    
    if(yin == y):
        maxContinuousCount += 1
    else:
        #Updating weights
        w[0] = w[0] + (y * alpha)
        w[1] = w[1] + (x[1] * y * alpha)
        w[2] = w[2] + (x[2] * y * alpha)
        w[3] = w[3] + (x[3] * y * alpha)
        w[4] = w[4] + (x[4] * y * alpha)
        maxContinuousCount = 0
    
    loopIterator = (loopIterator + 1)%1000
    numberOfIterations += 1

print("Obtained weights after training are : ")
print("b  = ", w[0])
print("w1 = ", w[1])
print("w2 = ", w[2])
print("w3 = ", w[3])
print("w4 = ", w[4])


"""
Testing on the same training set to determine wheather model fits or not.
"""
loopIterator = 0
yout = []
while(loopIterator < 1000):
    x[0] = 1
    x[1] = dt.ix[loopIterator]["X1"]
    x[2] = dt.ix[loopIterator]["X2"]
    x[3] = dt.ix[loopIterator]["X3"]
    x[4] = dt.ix[loopIterator]["X4"]
    
    yin = np.dot(w, x)
    if(yin >= theta):
        yout.append(1)
    elif(yin <= (-1 * theta)):
        yout.append(-1)
    else:
        yout.append(0)  
    loopIterator += 1

#Getting the accuracy of predicted output
from sklearn.metrics import accuracy_score
print("\nAcuracy on the training data : ")
print(accuracy_score(dt["Y"], yout))

print("\nNumber of nodes incorrectly classified : ")
print((1 - accuracy_score(dt["Y"], yout)) * 1000)

print("\nNumber of Iterations : ", numberOfIterations)