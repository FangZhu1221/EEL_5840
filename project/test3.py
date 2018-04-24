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
training = np.zeros([training_length,column*row])
testing = np.zeros([testing_length,column*row])
validation = np.zeros([validation_length,column*row])
for i in range(training_length):
    for j in range(row):
        for k in range(column):
            training[i,j*row+k] = training_images[i,j,k]
for i in range(validation_length):
    for j in range(row):
        for k in range(column):
            validation[i,j*row+k] = validation_images[i,j,k]
for i in range(testing_length):
    for j in range(row):
        for k in range(column):
            testing[i,j*row+k] = testing_images[i,j,k]
print(validation.shape)
print(testing.shape)
print(training.shape)
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

def sig(x):
    row = x.shape[0]
    col = x.shape[1]
    t = np.zeros([row,col])
    for i in range(row):
        for j in range(col):
            t[i,j] = (1/(1+np.exp(-x[i,j])))
    return t


def der_sig(x):
    row = x.shape[0]
    col = x.shape[1]
    t = np.zeros([row,col])
    for i in range(row):
        for j in range(col):
            t[i,j] = (1/(1+np.exp(-x[i,j])))*(1-(1/(1+np.exp(-x[i,j]))))
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
new_dimensions = 100
batch_size = 500
hidden_dimensions = 300
output_dimensions = 10
length = training_length
Bias_Input = np.ones([batch_size,1])
new_training = PCA(training/255,new_dimensions)
labels = np.zeros([length,output_dimensions])
for i in range(length):
    index = round(training_labels[i])
    labels[i,index] = 1
iteration = 5
step1 = 0.01
step2 = 0.01
J = np.zeros([output_dimensions,iteration])

w1 = np.random.uniform(0,1,(new_dimensions+1,hidden_dimensions))#uniform
w2 = np.random.uniform(0,1,(hidden_dimensions+1,output_dimensions))


for i in range(iteration):
    """
    old_order = np.array(np.arange(0,length,1))
    new_order = np.zeros(length)
    new_order = random.sample(old_order.tolist(),length)
    """
    mini_t = np.zeros([batch_size,new_dimensions])
    mini_l = np.zeros([batch_size,output_dimensions])
    for k in range(int(length/batch_size)):
        mini_t = new_training[k*batch_size:(k+1)*batch_size,:]
        mini_l = labels[k*batch_size:(k+1)*batch_size,:]

        Input = np.asmatrix(Bias_Input)
        for j in range(new_dimensions):
            Input = np.c_[Input,mini_t[:,j]] 
        # forward
        net1 = np.dot(Input,w1)
        hd_output = ReLu(net1)
        hd_input = np.c_[Bias_Input,hd_output]
    
        net2 = np.dot(hd_input,w2)
        Output = ReLu(net2)
        
        error = np.array(mini_l-Output)
    
        # backpropagation
        der_output = der_ReLu(net2)
        delta_output = np.multiply(error,der_output)
        D2 = np.dot(hd_input.T,delta_output)
        
        der_hidden = der_ReLu(net1)
        delta_hidden = delta_output@w2[1:hidden_dimensions+2,:].T
        D1 = Input.T@np.multiply(delta_hidden,der_hidden)
        #update
        w1 = w1 + step1*D1/batch_size
        w2 = w2 + step2*D2/batch_size
    #for k in range(output_dimensions):
        #J[k,i] = 0.5*(mini_l[:,k]-Output[:,k]).T@(mini_l[:,k]-Output[:,k])/batch_size
#    J[i] = M.sum(axis = 1).sum(axis = 0)

t = np.arange(0,iteration,1)    
plt.plot(t,J[0,:])    
        
