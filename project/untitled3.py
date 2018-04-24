# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 00:12:47 2017

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
validation = np.reshape(testing_images,(testing_length,column*row))

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

def sig(x):
    t = (1/(1+np.exp(-x)))
    return t


def der_sig(x):
    t = x*(1-x)
    return t

#%%
def forward(data,W1,W2):
    Input = np.zeros([10000,1])
    for i in range(10000):
        Input[i,0] = 1
    net1 = np.dot(data,W1)
    output1 = sig(net1)
    #output1 = ReLu(net1)
    Output1 = np.column_stack([Input,output1])
    net2 = np.dot(Output1,W2)
    output2 = sig(net2)
    return output2
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
new_dimensions = 784
batch_size = 500
batch_iteration = int(training_length/batch_size)
hidden_dimensions = 15
Bias_Input = np.ones([training_length,1])
Bias = np.ones([batch_size,1])
#new_training = PCA(training/255,new_dimensions)
new_training = training/255
iteration = 200
step1 = 0.01
step2 = 0.01
J = np.zeros([output_dimensions,iteration])
w1 = np.random.randn(new_dimensions+1,hidden_dimensions)
w2 = np.random.randn(hidden_dimensions+1,output_dimensions)
ArrayInput = np.column_stack([new_training,Bias_Input])


for i in range(iteration):
    for j in range(batch_iteration):
        data = ArrayInput[batch_size*j:batch_size*(j+1),:]
        dataOutput_training = tr_labels[batch_size*j:batch_size*(j+1),:]
        net1 = np.dot(data,w1)
        output1 = sig(net1)
        #output1 = ReLu(net1)
        Output1 = np.column_stack([Bias,output1])
        net2 = np.dot(Output1,w2)
        output2 = sig(net2)
        
        error_output2 = dataOutput_training-output2
        de_output2 = np.array(der_sig(output2))
        delta3 = error_output2*de_output2
        dJdW2 = (Output1.T@delta3)/batch_size
        
        delta2 = delta3@w2[1:hidden_dimensions+1,:].T
        #de_output1 = np.array(ReLuprime(net1))
        de_output1 = np.array(der_sig(output1))
        local_error = de_output1*delta2
        dJdW1 = (data.T@local_error)/batch_size
        
        w2 = w2 + 0.01*dJdW2
        w1 = w1 + 0.01*dJdW1


#new_validation = PCA(validation/255,new_dimensions)
new_validation = validation/255
Bias = np.ones([validation_length,1])
data = np.column_stack([Bias,new_validation])
Output = forward(data,w1,w2)
answer = np.argmax(Output, axis=1)
count = 0
for i in range(validation_length):
    if answer[i] == validation_labels[i]:
        count = count + 1
rate = count/validation_length
print(rate)
