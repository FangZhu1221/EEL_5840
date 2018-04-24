# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 21:13:12 2017

@author: JSZJZ
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 21:01:28 2017

@author: JSZJZ
"""

import numpy as np
import matplotlib.pyplot as plt
import math 
import copy
import time
import LoadMNIST as mn
import random

# load data
images,labels = mn.load_mnist(dataset="training", digits=np.arange(10), path="", size = 60000)
testing_images,testing_labels = mn.load_mnist(dataset="testing", digits=np.arange(10), path="", size = 10000)
# cross-validation
training_images = images[0:50000,:,:]
validation_images = images[50000:,:,:]
training_labels = labels[0:50000]
validation_labels = labels[50000:]
# get length
row = training_images.shape[1]
column = training_images.shape[2]
training_length = training_images.shape[0]
testing_length = testing_images.shape[0]
validation_length = validation_images.shape[0]
# transfer 3-dimensions to 2-dimensions
training = np.reshape(training_images,(training_length,column*row))
testing = np.reshape(validation_images,(validation_length,column*row))
validation = np.reshape(testing_images,(validation_length,column*row))

output_dimensions = 10

tr_labels = np.zeros([training_length,output_dimensions])
for i in range(training_length):
    index = round(training_labels[i])
    tr_labels[i,index] = 1
    
va_labels = np.zeros([validation_length,output_dimensions])
for i in range(validation_length):
    index = round(validation_labels[i])
    va_labels[i,index] = 1
    
te_labels = np.zeros([testing_length,output_dimensions])
for i in range(testing_length):
    index = round(testing_labels[i])
    te_labels[i,index] = 1
print(validation.shape)
print(testing.shape)
print(training.shape)

#%%
# sigma ReLu
def ReLu(x):
    row = x.shape[0]
    col = x.shape[1]
    t = np.zeros([row,col])
    for i in range(row):
        for j in range(col):
            if x[i,j] <= 0:
                    t[i,j] = 0
            else:
                if x[i,j] > 0:
                    t[i,j] = x[i,j]
    return t
# derivative sigma ReLu
def der_ReLu(x):
    row = x.shape[0]
    col = x.shape[1]
    t = np.zeros([row,col])
    for i in range(row):
        for j in range(col):
            if x[i,j] <= 0:
                    t[i,j] = 0
            else:
                if x[i,j] > 0:
                    t[i,j] = 1
    return t

#%%
# PCA
def PCA(data, reserved_dimension):
    length = data.shape[0]
    dimension = data.shape[1]
    E = np.zeros(dimension) 
    Mx = np.zeros([length,dimension]) 
    eigen_vals = np.zeros(dimension)
    eigen_vecs = np.zeros([dimension,dimension])
    for i in range(dimension):
        E[i] = sum(data[:,i])/length
    for i in range(dimension):
        Mx[:,i] = np.array([(data[m,i]-E[i]) for m in range(length)])        
    Cov = Mx.T@Mx/(length)
    eigen_val, eigen_vec = np.linalg.eig(Cov)
    for i in range(dimension):
        eigen_vals[i] = np.real(eigen_val[i]) 
    for i in range(dimension):
        for j in range(dimension):
            eigen_vecs[i,j] = np.real(eigen_vec[i,j])
    eigen_pairs = [(np.abs(eigen_vals[i]), np.array(eigen_vecs[:,i])) for i in range(len(eigen_vals))]
    eigen_pairs.sort(key = lambda x : x[0],reverse=True)
    w = eigen_pairs[0][1]
    for i in range(reserved_dimension-1):
        w = np.c_[w,eigen_pairs[i+1][1]]
    M1_pca = Mx@w
    return M1_pca

#%%
def transfer(x,w1,w2):
    length = x.shape[0]
    dimensions = x.shape[1]
    Bias_Input = np.ones([length,1])
    Input = np.asmatrix(Bias_Input)
    for j in range(dimensions):
        Input = np.c_[Input,x[:,j]] 
    net1 = np.dot(Input,w1)
    hd_output = ReLu(net1)
    hd_input = np.c_[Bias_Input,hd_output]

    net2 = np.dot(hd_input,w2)
    Output = ReLu(net2)
    return Output

#%%
print(w1)
print(w2)
length = validation_length
new_dimensions = 100
hidden_dimensions = 15
new_validation = PCA(validation/255,new_dimensions)
an = np.zeros([validation_length,10])
an = transfer(new_validation,w1,w2)
answer = np.zeros([validation_length,1])
for i in range(length):
    index = np.argmax(an[i,:])
    answer[i,0] = index
count = 0
for i in range(validation_length):
    if answer[i,0] == validation_labels[i]:
        count = count + 1
rate = count/length
print(rate)


