# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 11:33:04 2017

@author: Xi Yu
"""

import numpy as np
import os
import struct
import matplotlib
import matplotlib.pyplot as plt  
import math 
import textwrap
from array import array as pyarray
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import LoadMNIST as mn


Dataset_path = '/Users/Xi Yu/Desktop/machine learning_homework/project2/'

#sc      = StandardScaler()
# Load training data
training_images, training_labels  = mn.load_mnist(dataset="training", digits=np.arange(10), path="", size = 60000)
#t = training_images[5,:,:]
rows = training_images.shape[1]
cols = training_images.shape[2]
#plt.imshow(t, cmap ='Greys_r')
train_label = np.array(training_labels)
n_images = len(training_images)
n_labels = len(training_labels)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
train_image = training_images.reshape((n_images, -1))
train_labels = train_label.reshape((n_labels, -1))
features = train_image[:-1]
labels = train_labels[:-1]

Data = np.column_stack([train_image,train_labels])
# the number of the perceptron element of each layer
inputLayerSize = 784 
hiddenLayerSize = 15
outputLayerSize = 10
mini_batchsize = 500
min_batch = np.int(50000/mini_batchsize)

def ReLu(x):
        return np.maximum(x, 0)

def ReLuprime(x):
    return (x>0).astype(x.dtype)
    
#Sigmoid function
def sigmoid(x):
        return 1/(1+np.exp(-x))
   
#Gradient of sigmoid
def d_sig(x):
    return x*(1-x)

    
# the random initial value of weight and bias
W1 = np.random.randn(inputLayerSize+1,hiddenLayerSize) #add a bais
W2 = np.random.randn(hiddenLayerSize+1,outputLayerSize)


def label(train_label, label_size):
    desired = np.zeros(10)
    for i in range(label_size):
        if train_labels[i]==0:
            desired_lable = np.array([1,0,0,0,0,0,0,0,0,0])
        if train_labels[i]==1:
            desired_lable = np.array([0,1,0,0,0,0,0,0,0,0])
        if train_labels[i]==2:
            desired_lable = np.array([0,0,1,0,0,0,0,0,0,0])
        if train_labels[i]==3:
            desired_lable = np.array([0,0,0,1,0,0,0,0,0,0])
        if train_labels[i]==4:
            desired_lable = np.array([0,0,0,0,1,0,0,0,0,0])
        if train_labels[i]==5:
            desired_lable = np.array([0,0,0,0,0,1,0,0,0,0])
        if train_labels[i]==6:
            desired_lable = np.array([0,0,0,0,0,0,1,0,0,0])
        if train_labels[i]==7:
            desired_lable = np.array([0,0,0,0,0,0,0,1,0,0])
        if train_labels[i]==8:
            desired_lable = np.array([0,0,0,0,0,0,0,0,1,0])
        if train_labels[i]==9:
            desired_lable = np.array([0,0,0,0,0,0,0,0,0,1])
        desired = np.row_stack([desired,desired_lable])
    Data_output = desired[1:label_size+1,:].T
    return Data_output


def forward(data,W1,W2):
    Input = np.zeros([10000,1])
    for i in range(10000):
        Input[i,0] = 1
    net1 = np.dot(data,W1)
    output1 = sigmoid(net1)
    #output1 = ReLu(net1)
    Output1 = np.column_stack([Input,output1])
    net2 = np.dot(Output1,W2)
    output2 = sigmoid(net2)
    return output2

    
 
    
Input = np.zeros([50000,1])
for i in range(50000):
    Input[i,0] = 1


validation_Input = np.zeros([10000,1])
for i in range(10000):
    validation_Input[i,0] = 1
    
    
mini_input = np.zeros([mini_batchsize,1])
for i in range(mini_batchsize):
    mini_input[i,0] = 1


#pca = PCA(n_components=196, whiten=True)
#data = pca.fit_transform(DataInput_training)



#for i in range(30):
#np.random.shuffle(Data)
DataInput = Data[:,0:784]
DataOutput = Data[:,784]
DataInput_training = DataInput[0:50000,:]
Data_output = label(DataOutput,60000)
DataOutput_training = Data_output[:,0:50000].T
b = DataInput_training/np.max(DataInput_training) 
ArrayInput = np.column_stack([Input,b])
#arrayInput = np.array(ArrayInput,dtype=np.int16)

for i in range(20):
    for j in range(min_batch):
        data = ArrayInput[mini_batchsize*j:mini_batchsize*(j+1),:]
        dataOutput_training = DataOutput_training[mini_batchsize*j:mini_batchsize*(j+1)]
        net1 = np.dot(data,W1)
        output1 = sigmoid(net1)
        #output1 = ReLu(net1)
        Output1 = np.column_stack([mini_input,output1])
        net2 = np.dot(Output1,W2)
        output2 = sigmoid(net2)
        
        error_output2 = dataOutput_training-output2
        de_output2 = np.array(d_sig(output2))
        delta3 = error_output2*de_output2
        dJdW2 = (Output1.T@delta3)/mini_batchsize
        
        delta2 = delta3@W2[1:hiddenLayerSize+1,:].T
        #de_output1 = np.array(ReLuprime(net1))
        de_output1 = np.array(d_sig(output1))
        local_error = de_output1*delta2
        dJdW1 = (data.T@local_error)/mini_batchsize
        
        W2 = W2 + 0.01*dJdW2
        W1 = W1 + 0.01*dJdW1


# test in the validation data
validation_data = train_image[50000:60000,:]
validation_Data = validation_data/np.max(validation_data)
validation_d = np.column_stack([validation_Input,validation_Data])
DataOutput = train_label[50000:60000]
output = forward(validation_d, W1,W2)
predict = np.argmax(output, axis=1)
count = 0
for i in range(10000):
    if predict[i]==DataOutput[i]:
        count = count+1
accuracy = count/10000      
        
        
        
        
        
        
        
        
