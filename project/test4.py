# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 21:13:12 2017

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
validation = np.reshape(validation_images,(validation_length,column*row))
testing = np.reshape(testing_images,(testing_length,column*row))

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
    return np.maximum(x, 0)
    
# derivative sigma ReLu
def der_ReLu(x):
    return (x>0).astype(x.dtype)

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
new_dimensions = 784
batch_size = 20
batch_iteration = int(training_length/batch_size)
hidden_dimensions = 40
Bias_Input = np.ones([batch_size,1])
#new_training = PCA(training/255,new_dimensions)
new_training = training/255
iteration = 100
step1 = 0.1
step2 = 0.1
J = np.zeros([output_dimensions,iteration])
w1 = np.random.randn(new_dimensions+1,hidden_dimensions)
w2 = np.random.randn(hidden_dimensions+1,output_dimensions)

for i in range(iteration):
    mini_t = np.zeros([batch_size,new_dimensions])
    mini_l = np.zeros([batch_size,output_dimensions])
    for k in range(batch_iteration):
        old_order = np.array(np.arange(0,batch_size,1))
        new_order = np.zeros(batch_size)
        new_order = random.sample(old_order.tolist(),batch_size)
        for j in range(batch_size):
            index = int(new_order[j])
            mini_t[j,:] = new_training[k*batch_size+index,:]
            mini_l[j,:] = tr_labels[k*batch_size+index,:]
        
        Input = np.column_stack([Bias_Input,mini_t])
        # forward
        net1 = np.dot(Input,w1)
        hd_output = sig(net1)
        hd_input = np.column_stack([Bias_Input,hd_output])
    
        net2 = np.dot(hd_input,w2)
        Output = sig(net2)        
        error = np.array(mini_l-Output)
    
        # backpropagation
        der_output = der_sig(net2)
        delta_output = error*der_output
        D2 = (np.dot(hd_input.T,delta_output))/batch_size
        
        der_hidden = der_sig(net1)
        local_error = delta_output@w2[1:hidden_dimensions+1,:].T
        D1 = (Input.T@(local_error*der_hidden))/batch_size
        #update
        w1 = w1 + step1*D1
        w2 = w2 + step2*D2
    for k in range(output_dimensions):
        J[k,i] = 0.5*(mini_l[:,k]-Output[:,k]).T@(mini_l[:,k]-Output[:,k])/batch_size

#new_validation = PCA(validation/255,new_dimensions)
new_validation = validation/255
Bias = np.ones([validation_length,1])
put = np.column_stack([Bias,new_validation])
Net1 = np.dot(put,w1)
hd_output = sig(Net1)
hd_input = np.column_stack([Bias,hd_output])

Net2 = np.dot(hd_input,w2)
Out = sig(Net2)
answer = np.argmax(Out, axis=1)
count = 0
for i in range(validation_length):
    if answer[i] == validation_labels[i]:
        count = count + 1
rate = count/validation_length
print(rate)
t = np.arange(0,iteration,1)    
plt.plot(t,J[0,:]) 
plt.plot(t,J[1,:])
plt.plot(t,J[2,:])
plt.plot(t,J[3,:])
plt.plot(t,J[4,:])
plt.plot(t,J[5,:])
plt.plot(t,J[6,:])
plt.plot(t,J[7,:])
plt.plot(t,J[8,:])
plt.plot(t,J[9,:])