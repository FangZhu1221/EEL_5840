# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:31:10 2017

@author: JSZJZ
"""

# Relu and Sigmoid # learning curve (speed)
# learning cuve find the step size and mini-batch size (learning curve)
# PCA and downsample - accuarcy and confused matrix
# dropout for overfitting (training and validation) - learning curve and accuarcy
# SGD and momentum for the local optima - accuarcy and matrix
# output confused matrix and accuracy for the test datq - matrix and accuarcy


import numpy as np
import math
import matplotlib.pyplot as plt
import LoadMNIST as mn
import random
from sklearn.decomposition import PCA 


#%%
new_row = 14
new_col = 14
new_dimensions = new_row*new_col
shrink = int(math.sqrt(784/new_dimensions))
batch_size = 100
hidden_dimensions = 100
output_dimensions = 10
iteration = 400
step = 0.03
moment = 0.005
D1_before = 0
D2_before = 0
choice = 2
dropout = 0
mo = 1

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
batch_iteration = int(training_length/batch_size)
Bias_Input = np.ones([batch_size,1])

w1 = np.random.randn(new_dimensions+1,hidden_dimensions)*0.1
w2 = np.random.randn(hidden_dimensions+1,output_dimensions)*0.1
fig = plt.figure(figsize = (10,60))  

#%%
def ChangeLabelstoDec(labels,length):
    t = np.zeros([length,10])
    for i in range(length):
        index = round(labels[i])
        t[i,index] = 1
    return t

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
def MLP_Learning(x,l,activation,step,dropout,momentum):
    global w1
    global w2
    global D1_before
    global D2_before
    global moment
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
    
    if momentum == 0:
        moment = 0
    w1 = w1 + step1*D1 - moment*D1_before
    w2 = w2 + step2*D2 - moment*D2_before
    D1_before = step1*D1
    D2_before = step2*D2
    return error

#%%
def downsample(myarr,factor):
    ys,xs = myarr.shape
    crarr = myarr[:ys-(ys % int(factor)),:xs-(xs % int(factor))]
    dsarr = np.mean( np.concatenate([[crarr[i::factor,j::factor] 
        for i in range(factor)] 
        for j in range(factor)]), axis=0)
    return dsarr


#%%
# data process
if new_dimensions == 784:
    training = np.reshape(training_images,(training_length,column*row))
    validation = np.reshape(validation_images,(validation_length,column*row))
    testing = np.reshape(testing_images,(testing_length,column*row))
    tr_labels = ChangeLabelstoDec(training_labels,training_length)   
    va_labels = ChangeLabelstoDec(validation_labels,validation_length)     
    te_labels = ChangeLabelstoDec(testing_labels,testing_length)
    p1 = fig.add_subplot(*[6,1,1])
    p1.imshow(training_images[0,:,:], cmap='gray')
    new_training = training/255
    new_validation = validation/255
    new_testing = testing/255
else:
    if choice == 1:
        training = np.array(np.reshape(training_images,(training_length,column*row)))
        validation = np.array(np.reshape(validation_images,(validation_length,column*row)))
        testing = np.reshape(testing_images,(testing_length,column*row))
        tr_labels = ChangeLabelstoDec(training_labels,training_length)   
        va_labels = ChangeLabelstoDec(validation_labels,validation_length)     
        te_labels = ChangeLabelstoDec(testing_labels,testing_length)
        whole_data = np.row_stack([training,validation])
        pca = PCA(n_components=new_dimensions,whiten="true")
        whole_pca = pca.fit_transform(whole_data/255)
        new_training = whole_pca[0:50000,:]
        new_validation = whole_pca[50000:,:]
        display = np.around(np.reshape(new_training[0,:],(new_row,new_col))*255)
        p1 = fig.add_subplot(*[6,1,1])
        p1.imshow(display, cmap='gray')
    else:
        if choice == 2:
            new_training = np.zeros([training_length,new_row,new_col])
            new_validation = np.zeros([validation_length,new_row,new_col])
            new_testing = np.zeros([testing_length,new_row,new_col])
            for i in range(training_length):
                g = downsample(np.reshape(training_images[i,:,:],(row,column)),shrink)
                new_training[i,:,:] = np.reshape(g,(1,new_row,new_col))
            for i in range(validation_length):
                g = downsample(np.reshape(validation_images[i,:,:],(row,column)),shrink)
                new_validation[i,:,:] = np.reshape(g,(1,new_row,new_col))
            for i in range(testing_length):
                g = downsample(np.reshape(testing_images[i,:,:],(row,column)),shrink)
                new_testing[i,:,:] = np.reshape(g,(1,new_row,new_col))
            tr_labels = ChangeLabelstoDec(training_labels,training_length)   
            va_labels = ChangeLabelstoDec(validation_labels,validation_length)     
            te_labels = ChangeLabelstoDec(testing_labels,testing_length)
            p1 = fig.add_subplot(*[6,1,1])
            p1.imshow(new_training[0,:,:], cmap='gray')
            new_training = np.array(np.reshape(new_training,(training_length,new_col*new_row)))/255
            new_validation = np.array(np.reshape(new_validation,(validation_length,new_col*new_row)))/255
            new_testing = np.reshape(testing_images,(testing_length,column*row))/255
                    
#%%
p2 = fig.add_subplot(*[6,1,2])
b = np.zeros(10)
t = np.arange(0.002,0.022,0.002)
for m in range(10):
    moment = 0.002*m + 0.002
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
            error = MLP_Learning(mini_t,mini_l,1,step,dropout,mo)   
    Out = MLP(new_validation,1,dropout)
    answer = np.argmax(Out, axis=1)
    count = 0
    for j in range(validation_length):
        if answer[j] == validation_labels[j]:
            count = count + 1
    accuracy = count/validation_length
    print(accuracy)
    b[m] = accuracy
p2.plot(t,b)