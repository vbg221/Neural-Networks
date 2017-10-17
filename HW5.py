# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 23:08:26 2017

Name : Vatsal Gopani
N ID : N17368916
NetID: vbg221

Description : Assignment 5 in Neural Network.
              Testing the use of Backpropagation by solving XOR problem.
              [1 Hidden Layer is used]

@author: vbg22
"""

import numpy as np
import math

"""
Functions sigmoid(), sigmoid_prime(), tanh(), tanh_prime()
Input  : x

Output : value computed by equations

Description : output based on the definition of the particular function.
              all the functions are defined as their standard definitions.
"""
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return (sigmoid(x)*(1.0-sigmoid(x)))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2

"""
NeuralNetwork class that takes care of all the functionality.
"""
class NeuralNetwork:

    def __init__(self, layers, activation='tanh'):
        """
        Set activation andactivation_prime functions
        """
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        """
        Set initial weights
        """
        self.weights = []
        for i in range(1, len(layers) - 1):
            r = np.random.random((layers[i-1] + 1, layers[i] + 1)) - 0.5
            self.weights.append(r)

        r = np.random.random( (layers[i] + 1, layers[i+1])) - 0.5
        self.weights.append(r)
        

    """
    function fit()
    Input  : training vector X
             target vector y
             learning rate (default value is 0.02)
             epochs (default value is 10000)
    """
    def fit(self, X, y, learning_rate=0.02, epochs=10000):
        
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        X = X.astype(int)
        
        
        
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)
                    
            # output layer
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            # we need to begin at the second to last layer 
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)
                *self.activation_prime(a[l]))

            #reverse
            #[level3(output)->level2(hidden)]=>[level2(hidden)->level3(output)]
            deltas.reverse()

            """
            backpropagation
            1. Multiply its output delta and input activation 
               to get the gradient of the weight.
            2. Subtract a ratio (percentage) of the gradient from the weight.
            """
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

            n = int(math.ceil(epochs / 2000.0)) * 100
            if k % n == 0:
                print ("Total Square Error after %s epochs : %s"
                       %(k, error[0]**2))

    def predict(self, x): 
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)      
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        #a = round(a, 4)
        return a


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Following code segment acts as main() function
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
Defining Neural Network and Input array and Target output.
"""
nn = NeuralNetwork([2,2,1])
X = np.array([[-1, -1],
              [-1, 1],
              [1, -1],
              [1, 1]])
y = np.array([-1, 1, 1, -1])

#fits the neural network on given input data
nn.fit(X, y, learning_rate = 0.05, epochs = 8000)

print("\nTesting : ")
#gives the output based on the input data
for e in X:
    print(e,nn.predict(e))