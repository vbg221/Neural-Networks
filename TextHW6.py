# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:01:31 2017

@author: vbg22
"""
#from random import seed
#from random import random
import numpy as np
import pandas as pd
#import sklearn 





#Calculates accuracy of your models output.
#solutions: model predictions as a list or numpy array
#real: model labels as a list or numpy array
#Return: number between 0 and 1 representing your model's accuracy
def evaluate(solutions, real):
     predictions = np.array(solutions)
     labels = np.array(real)
     return (predictions == labels).sum() / float(labels.size)

totalK = 0
ncCount= 0


class MLP:
    #Initialize method called when the object is first created
    def __init__(self, n0, n1, n2, x0, t):
        #Initializes the necessary values for the (n0,n1, n2) Multi Layer Perceptron Network
        self.layers = 2
        self.weights = []
        self.bias = []
        self.x0 = x0
        self.t = t
        self.error = float("inf")

        self.numberOfNeurons = []
        self.numberOfNeurons.append(n0)
        self.numberOfNeurons.append(n1)
        self.numberOfNeurons.append(n2)

        self.alpha = 0.2
        
        self.neurons = []
        self.neurons.append(np.asmatrix(np.full((1,n0), 0.00000000000)))
        self.neurons.append(np.asmatrix(np.full((1,n1), 0.00000000000)))
        self.neurons.append(np.asmatrix(np.full((1,n2), 0.00000000000)))
        
        self.s = []
        self.s.append(np.asmatrix(np.full((n1,1), 0.00000000000)))
        self.s.append(np.asmatrix(np.full((n2,1), 1.00000000000)))
        #print self.s[1]
        
        self.n = []
        self.n.append(np.asmatrix(np.full((1,n1), 0)))
        self.n.append(np.asmatrix(np.full((1,n2), 0)))
        
        self.weights.append(np.asmatrix(np.random.rand(n0, n1))-0.5)
        self.bias.append(np.asmatrix(np.random.rand(1, n1))-0.5)
        if(self.t == 1.0):
            self.weights[0] = self.weights[0] * 2
            self.bias[0] = self.bias[0] * 2
        if(self.t == 1.5):
            self.weights[0] = self.weights[0] * 2 * 1.5
            self.bias[0] = self.bias[0] * 2 * 1.5


        #self.bias[0] = np.asmatrix([[-0.3378, 0.2771, 0.2859, -0.3329]])
        #self.weights[0] = np.asmatrix([[0.1970, 0.3191, -0.1448, 0.3594], [0.3099, 0.1904, -0.0347, -0.4861]])
        
        self.weights.append(np.asmatrix(np.random.rand(n1, n2))-0.5)
        self.bias.append(np.asmatrix(np.random.rand(1, n2))-0.5)
        if(self.t == 1.0):
            self.weights[1] = self.weights[1] * 2
            self.bias[1] = self.bias[1] * 2
        if(self.t == 1.5):
            self.weights[1] = self.weights[1] * 2 * 1.5
            self.bias[1] = self.bias[1] * 2 * 1.5
        
        #self.bias[1] = np.asmatrix([[-0.1401]])
        #self.weights[1] = np.asmatrix([[0.4919], [-0.2913], [-0.3979], [0.3581]])
        
        

    #Binary sigmoid transfer function
    def transferFun(self, x):
        return ((1 - np.exp(-x/self.x0))/(1 + np.exp(-x/self.x0)))
    
    #Differentiated Binary sigmoid transfer function
    def defTransferFun(self, x):
        return ((0.5/self.x0) * (1 + self.transferFun(x)) * (1 - self.transferFun(x)))

    #Train method to train the MLP classifier
    def train(self, features, labels):
        #st = time.time()
        
#        flag = True
        k = 0
#        while flag:
#        while (self.s[1] > 0.1) or (self.s[1] < -0.1): #  (time.time() - st) < 25.000
        error_counter = 0
        while (self.error > 0.05) and (k < 3000) :
#        while (time.time() - st) < 10.000:
            k += 1
            self.error = 0
            error_counter = 0
            #print "iteration : ", k
            for index, record in enumerate(features.values):
                
                self.neurons[0] = np.asmatrix(record)
                #evaluates the neurons to find the error in prediction
                for i in range(0, self.layers):
                    self.n[i] = (self.neurons[i] * self.weights[i]) + self.bias[i]
                    self.neurons[i+1] = self.transferFun(self.n[i])
                
                """
                The code segment written below is hardcoded for networks with 1 final output neuron only.
                For networks with different dimensions, below written code segment must be executed for all individual final output neuron.
                """
                #calculates the sensitivity of the final layer
                if labels[index]:
                    l = 1
                else:
                    l = -1
                self.s[1] = ((self.neurons[2] - l)) * self.defTransferFun(self.n[1])
                """
                Hardcoded segment ends here.
                """
                #print "Original Output : ", self.neurons[2]
                #print "Original Label : ", l
                self.error += (self.neurons[2] - l) ** 2
                if self.error < 0.98:
                    error_counter += 1
                #print "Squared Error : ", self.error
                #print "\n\n"
                #Calculates the sensitivity by backpropogating the sensitivity of final layer
                for j in range (0, self.numberOfNeurons[1]):
                    total = 0
                    for element in self.weights[1][j]:
                        total += (element * self.s[1])
                    y = self.defTransferFun(self.n[0][0, j]) * total
                    self.s[0][j, 0] = y[0, 0]
                
                #Updates the weights and bias based on the sensitivity calculated previously
                for i in range(0, self.layers):
                    self.weights[i] = self.weights[i] - (self.alpha * (self.neurons[i].T * self.s[i].T))
                    self.bias[i] = self.bias[i] - (self.alpha * self.s[i].T)
        if k == 3000:            #flag = False
            #print "Not Converged"
            global ncCount
            ncCount += 1
        else:
            print "Epoch : ", k
            global totalK
            totalK += k
            global score
            score.append(k)

#        print "Layer 1 :"
#        print self.bias[0]
#        print self.weights[0]
#        print "\nLayer 2 : "
#        print self.bias[1]
#        print self.weights[1]

    #Predict method obtain predicted labels for input feature set
    def predict(self, features):
        output = []
        temp = []
        for index, record in enumerate(features.values):
            out = []
            self.neurons[0] = np.asmatrix(record)
            for i in range(0, self.layers):
                self.n[i] = (self.neurons[i] * self.weights[i]) + self.bias[i]
                self.neurons[i+1] = self.transferFun(self.n[i])
            temp.append(self.neurons[2][0, 0])
            if self.neurons[2][0, 0] > 0.0:
                out.append(True)
            else:
                out.append(False)
            output.append(out)
        return np.asarray(output), temp



features = pd.DataFrame([[1,1],[1,-1],[-1,1],[-1,-1]])
labels = np.asarray(list([[False], [True], [True], [False]]))

#mlp = MLP(2, 4, 1)
#mlp.train(features, labels)
#target = labels
#predicted, temp = mlp.predict(features)
##print "\n", temp
#print "Accuracy : ", evaluate(predicted, target)

dic = []
print "For Quadratic Functions : "
print "#######################################################################################################"
print "\n"

l = [2, 4, 6, 8, 10]
x = [0.5, 1.0, 1.5]
tau = [0.5, 1.0, 1.5]
for ta in tau:
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    print "x               t : %d               x" %(ta)
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n"
    for x0 in x:

        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        print "x              x0 : %d               x" %(x0)
        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n"

        for t in l:
            print "For N1 : ", t
            print "\n"
            m = {}
            m['x0'] = x0
            m['t'] = ta
            m['n'] = t
            m['median'] = 0
            m['mean'] = 0
            score = []
            for counter in range(0,100):
                mlp = MLP(2, t, 1, x0, ta)
                mlp.train(features, labels)
                target = labels
                predicted, temp = mlp.predict(features)
                #print "\n", temp
                #print "Accuracy : ", evaluate(predicted, target)
                del(mlp)
            m['Not Converged'] = ncCount
            
            
            print "\nX0 : ", x0
            print "t : ", ta
            print "\n\nNumber of times neural network did not converge : ", ncCount
            if totalK == 0:
                print "Average epochs : 0"
                m['Average Epoch'] = 0
                m['median'] = 0
                m['mean'] = 0
            else:
                print "Average epochs : ", (totalK/(100 - ncCount))
                m['Average Epoch'] = (totalK/(100 - ncCount))
                m['median'] = round(np.mean(score), 1)
                m['mean'] = round(np.median(score), 1)
            
            dic.append(m)
            
            print "\n"
            print "=================================================================================================="
            print ""
            del(m)
            del(score)
            ncCount = 0
            totalK = 0

print "Acquired Results : \n"
print dic

print "\nFrom the above acquired results, it can be inferred that the average epochs it takes to converge the neural network are inversely proportional to the number of neurons in the hidden layer."
print "\nAlso, the possibility that a neural net converges, increases with the increase in number of neurons in hidden layer."
print "\n"

print "For the constant value of t : "
print "With smaller values of x0, the neural network converges faster with epochs usually in two digits only."
print "As the value of x0 increases, the neural network takes more time (and epochs) to converge"
print "\n"

print "For the constant value of x0 : "
print "With the smaller value of t, the neural network converges faster."
print "As the value of t increases, the neural network takes more time (and epochs) to converge"
print ""
print "=================================================================================================="




print "\n\nNow, for N1 : 1"
print "\n"
for i in range (0, 100):
    mlp = MLP(2, 1, 1, 1.0, 1.0)
    mlp.train(features, labels)
    target = labels
    predicted, temp = mlp.predict(features)
    #print "\n", temp
    #print "Accuracy : ", evaluate(predicted, target)
    del(mlp)

print "\n\nNumber of times neural network did not converge : ", ncCount
print "Average epochs : 0"#, (totalK/(100 - ncCount))
print "\n"
print "=================================================================================================="
print ""
ncCount = 0
totalK = 0

print "From the above acquired results, it can be inferred that the neural network with only one neuron in hidden layer does not converge for any non liniear problem."
print "\nWhich means, that Multi Layer Perceptron has no advantage over Simple Perceptron if there is only one neuron in hidden layer."


# =============================================================================
# for row in dataset:
# 	print("Row : ", row)
# 	prediction = predict(network, row)
# 	print('Expected=%d, Got=%.9f\n' % (row[-1], prediction[0]))
# =============================================================================
