# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 21:27:25 2017

Name : Vatsal Gopani
N ID : N17368916
NetID: vbg221

Description : Assignment 3.2 in Neural Network.
              Testing the use of Multiclass Perceptron. 

@author: vbg22
"""


import numpy as np
import pandas as pd


x = [[[1], [1]], [[1], [2]], [[2], [-1]], [[2], [0]], [[-1], [2]], [[-2], [1]], [[-1], [-1]], [[-2], [-2]]]
y = [[[-1], [-1]], [[-1], [-1]], [[-1], [1]], [[-1], [1]], [[1], [-1]], [[1], [-1]], [[1], [1]], [[1], [1]]]
a = [[], []]
e = [[], []]
maxContinuousCount = 0
loopIterator = 0

w = [[0, 0], [0, 0]]
b = [[0], [0]]
theta = 1
alpha = 1.5
numberOfIteration = 0

"""
Function hardlim()
input  : A column vector

Output : A column vector with same dimention as input vector

Description : This function takes input vector and checks each 
              element of the vector. 
              - If any element is greater then the theta, it gives 1. 
              - For element less then -theta, it gives -1. 
              - For the rest, it gives 0.
"""
def hardlim(a):
    b = [[0], [0]]
    if(a[0][0] > theta):
        b[0][0] = 1
    elif(a[0][0] < (-1 * theta)):
        b[0][0] = -1
    else:
        b[0][0] = 0
    
    if(a[1][0] > theta):
        b[1][0] = 1
    elif(a[1][0] < (-1 * theta)):
        b[1][0] = -1
    else:
        b[1][0] = 0
    return b

"""
Function checkZ()
inut   : A column vector

output : Boolean value

Description : This function checks whether the input vector has 
              all the elements zero elements or not. It returns 
              TRUE in case the whole vector is zero vector and 
              FALSE otherwise.
"""
def checkZ(e):
    if(e[0][0] == 0 and e[1][0] == 0):
        return True
    else:
        return False
    
"""
Training the Neurons so that weight can be adjusted and be used later on for prediction
"""
while(maxContinuousCount != 8):
    a = np.asmatrix(hardlim(np.asmatrix(np.add(np.matmul(w, x[loopIterator]), b))))
    e = np.asmatrix(np.add(y[loopIterator], np.asmatrix(np.dot(-1, a))))
    
    if(checkZ(e)):
        maxContinuousCount += 1
    else:
        w = np.asmatrix(np.add(w, np.asmatrix(np.dot(alpha, np.asmatrix(np.matmul(e, np.transpose(x[loopIterator])))))))
        b = np.asmatrix(np.add(b, np.asmatrix(np.dot(alpha, e))))
        
        maxContinuousCount = 0
    
    loopIterator = (loopIterator + 1) % 8
    numberOfIteration += 1


print("Alpha : ", alpha)
print("Theta : ", theta)
print("Weights : \n", w, "\n")
print("Bias : \n", b, "\n")

loopIterator = 0
correct = 0
incorrect = 0

"""
Testing on the same training set to determine wheather model fits or not.
"""
while(loopIterator != 8):
    a = np.asmatrix(hardlim(np.asmatrix(np.add(np.matmul(w, x[loopIterator]), b))))
    e = np.asmatrix(np.add(y[loopIterator], np.asmatrix(np.dot(-1, a))))
    
    if(checkZ(e)):
        correct += 1
    else:
        incorrect += 1

    loopIterator += 1

print("Number of nodes classified correctly : ", correct)
print("Number of nodes classified incorrectly : ", incorrect)