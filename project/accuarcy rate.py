# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 09:49:15 2017

@author: JSZJZ
"""


import numpy as np
import matplotlib.pyplot as plt
import LoadMNIST as mn
import random

new_dimensions = 784
batch_size = 100
hidden_dimensions = 0
output_dimensions = 10
iteration = 400
step = 0.02



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
def MLP(x,activation,dropout):
    global w1
    global w2
    xlength = x.shape[0]
    Bias = np.ones([xlength,1])
    if dropout == 1:
        w1 = 0.5*w1
        w2 = 0.5*w2
    Input = np.column_stack([Bias,x])
    Net1 = np.dot(Input,(w1))
    if activation == 0:
        hd_output = sig(Net1)
    else:
        if activation == 1:
            hd_output = ReLu(Net1)
    hd_input = np.column_stack([Bias,hd_output])
    Net2 = np.dot(hd_input,(w2))
    if activation == 0:
        Output = sig(Net2)
    else:
        if activation == 1:        
            Output = ReLu(Net2)
    return Output
    

#%%
def MLP_Learning(x,l,activation,step,dropout):
    global w1
    global w2
    step1 = step
    step2 = step
    size = x.shape[0]
    Bias_Input = np.ones([size,1])
    Input = np.column_stack([Bias_Input,x])
    p = 0.5
    P =  np.random.binomial([np.ones((size,hidden_dimensions))],p)[0]
    size = x.shape[0]
    Bias_Input = np.ones([size,1])
    Input = np.column_stack([Bias_Input,x])
    net1 = np.dot(Input,w1)
    if activation == 0:
        hd_output = sig(net1)
    else:
        if activation == 1:
            hd_output = ReLu(net1)
    if dropout == 1:
        hd_output = np.multiply(hd_output,P)
    hd_input = np.column_stack([Bias_Input,hd_output])
    
    net2 = np.dot(hd_input,w2)
    if activation == 0:
        Output = sig(net2) 
    else:
        if activation == 1:
            Output = ReLu(net2)       
    error = np.array(l-Output)
    
    # backpropagation
    if activation == 0:
        der_output = der_sig(net2)
    else:
        if activation == 1:
            der_output = der_ReLu(net2)       
    delta_output = error*der_output
    D2 = (np.dot(hd_input.T,delta_output))/batch_size
        
    if activation == 0:
        der_hidden = der_sig(net1)
    else:
        if activation == 1:
            der_hidden = der_ReLu(net1) 
    local_error = delta_output@w2[1:hidden_dimensions+1,:].T
    D1 = (Input.T@(local_error*der_hidden))/batch_size
    
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
# transfer 3-dimensions to 2-dimensions
training = np.reshape(training_images,(training_length,column*row))
validation = np.reshape(validation_images,(validation_length,column*row))
testing = np.reshape(testing_images,(testing_length,column*row))
output_dimensions = 10
tr_labels = ChangeLabelstoDec(training_labels,training_length)   
va_labels = ChangeLabelstoDec(validation_labels,validation_length)     
te_labels = ChangeLabelstoDec(testing_labels,testing_length)
print(validation.shape)
print(testing.shape)
print(training.shape)

batch_iteration = int(training_length/batch_size)
Bias_Input = np.ones([batch_size,1])
#new_training = PCA(training/255,new_dimensions)
new_training = training/255
a = np.zeros(iteration)
for m in range(2):
    hidden_dimensions = m*50 + 200
    w1 = np.random.randn(new_dimensions+1,hidden_dimensions)*0.1
    w2 = np.random.randn(hidden_dimensions+1,output_dimensions)*0.1
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
            MLP_Learning(mini_t,mini_l,1,step,0)
        new_validation = validation/255
        Out = MLP(new_validation,1,0)
        answer = np.argmax(Out, axis=1)
        count = 0
        for j in range(validation_length):
            if answer[j] == validation_labels[j]:
                count = count + 1
        accuracy = count/validation_length
        a[i] = accuracy
    t = np.arange(iteration)
    plt.plot(t,a,'r')

b = np.zeros(iteration)
for m in range(2):
    hidden_dimensions = m*10 + 20
    w1 = np.random.randn(new_dimensions+1,hidden_dimensions)*0.1
    w2 = np.random.randn(hidden_dimensions+1,output_dimensions)*0.1
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
            MLP_Learning(mini_t,mini_l,1,step,1)
        new_validation = validation/255
        Out = MLP(new_validation,1,1)
        answer = np.argmax(Out, axis=1)
        count = 0
        for j in range(validation_length):
            if answer[j] == validation_labels[j]:
                count = count + 1
        accuracy = count/validation_length
        b[i] = accuracy
    t = np.arange(iteration)
    plt.plot(t,b,'b')