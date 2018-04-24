# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 10:58:18 2017

@author: JSZJZ
"""


import numpy as np
import matplotlib.pyplot as plt
import LoadMNIST as mn
import random
from sklearn.decomposition import PCA 

batch_size = 100
hidden_dimensions = 15Q
output_dimensions = 10


#%%
def ChangeLabelstoDec(labels,length):
    t = np.zeros([length,10])
    for i in range(length):
        index = round(labels[i])
        t[i,index] = 1
    return t


#%%
# sigma ReLu
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
def downsample(myarr,factor):
    ys,xs = myarr.shape
    crarr = myarr[:ys-(ys % int(factor)),:xs-(xs % int(factor))]
    dsarr = np.mean( np.concatenate([[crarr[i::factor,j::factor] 
        for i in range(factor)] 
        for j in range(factor)]), axis=0)
    return dsarr


#%%
def MLP(x,activation = 0):
    global w1
    global w2
    if activation == 0:
        xlength = x.shape[0]
        Bias = np.ones([xlength,1])
        Input = np.column_stack([Bias,x])
        Net1 = np.dot(Input,w1)
        hd_output = sig(Net1)
        hd_input = np.column_stack([Bias,hd_output])
        Net2 = np.dot(hd_input,w2)
        Output = sig(Net2)
    else:
        if activation == 1:
            xlength = x.shape[0]
            Bias = np.ones([xlength,1])
            Input = np.column_stack([Bias,x])
            Net1 = np.dot(Input,w1)
            hd_output = ReLu(Net1)
            hd_input = np.column_stack([Bias,hd_output])
            Net2 = np.dot(hd_input,w2)
            Output = ReLu(Net2)
    return Output
    


#%%
def MLP_Learning(x,l,activation,step):
    global w1
    global w2
    step1 = step
    step2 = step
    if activation == 0:
        size = x.shape[0]
        Bias_Input = np.ones([size,1])
        Input = np.column_stack([Bias_Input,x])
        net1 = np.dot(Input,w1)
        hd_output = sig(net1)
        hd_input = np.column_stack([Bias_Input,hd_output])
    
        net2 = np.dot(hd_input,w2)
        Output = sig(net2)        
        error = np.array(l-Output)
    
        # backpropagation
        der_output = der_sig(net2)
        delta_output = error*der_output
        D2 = (np.dot(hd_input.T,delta_output))/batch_size
        
        der_hidden = der_sig(net1)
        local_error = delta_output@w2[1:hidden_dimensions+1,:].T
        D1 = (Input.T@(local_error*der_hidden))/batch_size
    else:
        if activation == 1:
            size = x.shape[0]
            Bias_Input = np.ones([size,1])
            Input = np.column_stack([Bias_Input,x])
            net1 = np.dot(Input,w1)
            hd_output = ReLu(net1)
            hd_input = np.column_stack([Bias_Input,hd_output])
    
            net2 = np.dot(hd_input,w2)
            Output = ReLu(net2)        
            error = np.array(l-Output)
    
            # backpropagation
            der_output = der_ReLu(net2)
            delta_output = error*der_output
            D2 = (np.dot(hd_input.T,delta_output))/batch_size
        
            der_hidden = der_ReLu(net1)
            local_error = delta_output@w2[1:hidden_dimensions+1,:].T
            D1 = (Input.T@(local_error*der_hidden))/batch_size
        #update
    w1 = w1 + step1*D1
    w2 = w2 + step2*D2
    return error


#%%
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
a = np.zeros(8)
t = np.arange(2,10,1)

for m in range(8):
    shrink = m + 2
    new_row = np.int(28/shrink)
    new_col = np.int(28/shrink)
    new_training = np.zeros([training_length,new_row,new_col])
    new_validation = np.zeros([validation_length,new_row,new_col])
    for i in range(training_length):
        g = downsample(np.reshape(training_images[i,:,:],(row,column)),shrink)
        new_training[i,:,:] = np.reshape(g,(1,new_row,new_col))
    for i in range(validation_length):
        g = downsample(np.reshape(validation_images[i,:,:],(row,column)),shrink)
        new_validation[i,:,:] = np.reshape(g,(1,new_row,new_col))
    # transfer 3-dimensions to 2-dimensions
    training = np.array(np.reshape(new_training,(training_length,new_col*new_row)))
    validation = np.array(np.reshape(new_validation,(validation_length,new_col*new_row)))
    testing = np.reshape(testing_images,(testing_length,column*row))
    output_dimensions = 10
    tr_labels = ChangeLabelstoDec(training_labels,training_length)   
    va_labels = ChangeLabelstoDec(validation_labels,validation_length)     
    te_labels = ChangeLabelstoDec(testing_labels,testing_length)

    new_dimensions = new_row*new_col
    w1 = np.random.randn(new_dimensions+1,hidden_dimensions)*0.1
    w2 = np.random.randn(hidden_dimensions+1,output_dimensions)*0.1

    batch_iteration = int(training_length/batch_size)
    Bias_Input = np.ones([batch_size,1])
    new_training = training/255
    new_validation = validation/255

    iteration = 100
    step = 0.01
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
            error = MLP_Learning(mini_t,mini_l,1,step)   
    Out = MLP(new_validation,1)
    answer = np.argmax(Out, axis=1)
    count = 0
    for j in range(validation_length):
        if answer[j] == validation_labels[j]:
            count = count + 1
    accuracy = count/validation_length
    a[m] = accuracy
    
plt.plot(t,a)
