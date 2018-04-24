# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:08:23 2017

@author: JSZJZ
"""
import numpy as np
import matplotlib.pyplot as plt
import math 
import copy
import time
import LoadMNIST as mn

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
    t = (1/(1+np.exp(-x)))
    return t


def der_sig(x):
    t = (1/(1+np.exp(-x)))*(1-(1/(1+np.exp(-x))))
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
new_dimensions = 64
hidden_dimensions = 512
output_dimensions = 4
new_training = PCA(training/255,new_dimensions)
"""
raw, column = new.shape
position1 = np.argmax(new)
position2 = np.argmin(new)
m1, n1 = divmod(position1, column)  
m2, n2 = divmod(position2, column) 
hu = (-new[m2,n2])*np.ones([50000,new_dimensions])
new_training = (PCA(training,new_dimensions)+hu)/(new[m1,n1]-new[m2,n2])
"""
iteration = 3
step1 = 0.005
step2 = 0.000005
J = np.zeros(iteration)

#w1 = 0.0001*abs(np.random.randn(new_dimensions+1,hidden_dimensions))
#w2 = 0.00001*abs(np.random.randn(hidden_dimensions+1,1))
w1 = 0.01*np.ones([new_dimensions+1,hidden_dimensions])
w2 = 0.001*np.ones([hidden_dimensions+1,output_dimensions])

print(w1)
print(w2)

for i in range(iteration):
    length = new_training.shape[0]
    dimensions = new_training.shape[1]
    hidden_dimensions = w2.shape[1]
    Bias_Input = np.ones([length,1])
    validation_labels = np.zeros([length,output_dimensions])
    v = np.zeros([length,1])
    validation_output = np.array(np.asmatrix(training_labels).T)
    Input = np.asmatrix(Bias_Input)
    for i in range(dimensions):
        Input = np.c_[Input,new_training[:,i]] 
    # forward
    net1 = np.dot(Input,w1)
    hd1_output = ReLu(net1)
    hd1_input = np.c_[Bias_Input,hd1_output]
    
    net2 = np.dot(hd1_input,w2)
    Output = sig(net2)
    # backward
    re = 0
    for i in range(length):
        v[i,0] = bin(int(validation_output[i,0]))
    for i in range(output_dimensions):
        validation_labels[:,i] = (v>>i)%2
    error = np.array(validation_labels-Output)
    der_output = der_sig(net2)
    delta_output = error*der_output
    DW2 = hd1_input.T@delta_output
    
    delta_hidden = delta_output*w2[1:hidden_dimensions+1].T
    der_hidden = der_ReLu(net1)
    DW1 = Input.T@(der_hidden*delta_hidden)
    # update
    w1 = w1 + step1*DW1
    w2 = w2 + step2*DW2
    J = 0.5*(validation_output-Output).T@(validation_output-Output)/length
        
