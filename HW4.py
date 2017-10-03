# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 17:43:33 2017

Name : Vatsal Gopani
N ID : N17368916
NetID: vbg221

Description : Assignment 4 in Neural Network.
              Testing the use of Multiclass Perceptron with ADALINE algorithm. 

@author: vbg22
"""

import numpy as np

"""
Initialize the given input and output data.
"""
x = [[[1], [1]], [[1], [2]], [[2], [-1]], [[2], [0]], [[-1], [2]], [[-2], [1]],
     [[-1], [-1]], [[-2], [-2]]]

y = [[[-1], [-1]], [[-1], [-1]], [[-1], [1]], [[-1], [1]], [[1], [-1]], 
     [[1], [-1]], [[1], [1]], [[1], [1]]]

"""
Initialize the weights and other parameters.
weight w and bias b is set to zero.
"""
alpha = 0.5
theta = 0.014
w = np.asmatrix([[0, 0], [0, 0]])
b = np.asmatrix([[0], [0]])

#Other parameters and variables.
a = np.asmatrix([[0], [0]])
e = np.asmatrix([[0], [0]])
loopIterator = 0
numberOfIteration = 0
yin = np.asmatrix([[0], [0]])

"""
Function activationFunction()
Input  : Weight w
         Input vector to neural network X(k)
         Bias b

Output : column vector with the same dimension 

Description : The function actually implements this equation.
              y = f(x) = Sigma(w*x) + b
"""
def activationFunction(w, x, b):
    initSum = np.asmatrix([[0],[0]])
    initSum = np.matmul(w, x)
    
    finalSum = np.asmatrix([[0], [0]])
    finalSum =  np.add(b, initSum)
    return finalSum

"""
Function testTransferFunction()
Input  : Column vector e with the dimensions (2 x 1)

Output : Column vector of the same dimensions as input.

Description : This function sets the output vector to values 
              depending on the input and theta provided.
              output = 1; (e >= theta)
                      -1; (e < theta)
"""
def testTransferFunction(e):
    val = np.asmatrix([[0],[0]])
    if(e[0][0] >= theta):
        val[0][0] = 1
    else:
        val[0][0] = -1
    if(e[1][0] >= theta):
        val[1][0] = 1
    else:
        val[1][0] = -1
    return val

"""
Function update()
Input  : Bias b
         Weight w
         Transpose(InputVector x)
         Output of activation function
         Target variable t
         
Output : Two vectors of dimensions (2 x 2) and (2 x 1)

Description : This function updates the weghts and bias by following delta rule
              w(new) = w(old) - (2 * alpha * x) * Sigma(y - t)
              b(new) = b(old) - (2 * alpha) * Sigma(y -t)
"""
def update(b, w, xt, yin, t):
    newW = np.asmatrix([[0, 0], [0, 0]])
    newb = np.asmatrix([[0], [0]])
    
    newW = np.add(w, np.dot((-2 * alpha), 
                            np.matmul(np.add(yin, np.dot(-1, t)), xt)))
    newb = np.add(b, np.dot((-2 * alpha), np.add(yin, np.dot(-1, t))))
    
    return newW, newb

"""
Funciton checkZ()
Input  : Column vector of dimension (2 x 1)

Output : boolean value depending on input

Description : Gives the boolean value in output.
              True; if input vector is zero vector
              False; if input vector is not zero vector
"""
def checkZ(e):
    if(e[0][0] == 0 and e[1][0] == 0):
        return True
    else:
        return False



"""
Train the Neurons based on the Adaline model of Neural Network.
Alpha can be updated or kept static depending on the choice of user.
"""
while(numberOfIteration != 5000):
    yin = activationFunction(w, np.asmatrix(x[loopIterator]), b)
    w, b = update(b, w, np.transpose(np.asmatrix(x[loopIterator])), 
                  yin, np.asmatrix(y[loopIterator]))   
    
    loopIterator = (loopIterator + 1) % 8
    numberOfIteration += 1
    #Comment this line below to keep alpha static
    alpha = 0.999 * alpha 
    #alpha = 1 / numberOfIteration   

"""
Print Obtained Weights and Bias for given Alpha
"""
print("Alpha : ", alpha)
print("\nw : \n", np.transpose(w))
print("\nb : \n", b)

"""
Resetting the variables.
"""
loopIterator = 0
correct = 0
incorrect = 0

"""
Test the data against the neurons with obtained weights and bias.
"""
while(loopIterator != 8):
    yin = activationFunction(w, np.asmatrix(x[loopIterator]), b)
    a = testTransferFunction(yin)
    
    e = np.add(np.asmatrix(y[loopIterator]), np.dot(-1, a))
    
    if(checkZ(e)):
        correct += 1
    else:
        incorrect += 1
    
    loopIterator += 1

"""
Print the output.
"""
print ("\nNumber of nodes classified correctly : ", correct)
print ("Number of nodes classified incorrectly : ", incorrect)