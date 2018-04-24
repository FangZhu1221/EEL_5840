# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:07:27 2017

@author: JSZJZ
"""


import numpy as np
import matplotlib.pyplot as plt
import math 
import time


#initial the parameters
w11 = np.random.uniform(size=(1, 1))
w21 = np.random.uniform(size=(1, 1))
w12 = np.random.uniform(size=(1, 1))
w22 = np.random.uniform(size=(1, 1))
w31 = np.random.uniform(size=(1, 1))
w32 = np.random.uniform(size=(1, 1))
w41 = np.random.uniform(size=(1, 1))
w42 = np.random.uniform(size=(1, 1))
w51 = np.random.uniform(size=(1, 1))
w52 = np.random.uniform(size=(1, 1))
b11 = np.random.uniform(size=(1, 1))
b12 = np.random.uniform(size=(1, 1))
b21 = np.random.uniform(size=(1, 1))
b22 = np.random.uniform(size=(1, 1))
b3 = np.random.uniform(size=(1, 1))
x21 = 0
x22 = 0
x31 = 0
x32 = 0
y = 0
t11 = 0
t12 = 0
t21 = 0
t22 = 0
t3 = 0
e = 0

step1 = 0.5
step2 = 0.5
step3 = 0.5
test_data = np.zeros(2)
test_data[0] = 0.32
test_data[1] = 0.92
#sigmoid = 1/(1+np.exp())
#load data
data = np.loadtxt("HW6_Data.txt")
validate = data[:,2]
training = data[:,0:2]
length = validate.size
track1 = np.zeros(length)
track2 = np.zeros(length)
track3 = np.zeros(length)
error = np.zeros(length)
x = np.zeros(400)
v = np.zeros(400)
    
def der_sig(x):
    return (1/(1+np.exp(-x)))*(1-(1/(1+np.exp(-x))))

for j in range(1000):
    for i in range(length):
        #calculate the answers for the output layer
        #input layer
        x11 = training[i,0]
        x12 = training[i,1]
        x13 = 1
        #hidden layer 1
        t11 = w11*x11 + w21*x12 + b11*x13
        t12 = w12*x11 + w22*x12 + b12*x13
        x21 = 1/(1+np.exp(-t11))
        x22 = 1/(1+np.exp(-t12)) 
        x23 = 1
        x[i] = x21
        v[i] = x22
        #hidden layer 2
        t21 = w31*x21 + w41*x22 + b21*x23
        t22 = w32*x21 + w42*x22 + b22*x23
        x31 = 1/(1+np.exp(-t21))
        x32 = 1/(1+np.exp(-t22))  
        x33 = 1
        #output layer
        t3 = w51*x31 + w52*x32 + b3*x33
        y = 1/(1+np.exp(-t3))
        
        #calculate the backpropagation 
        #output layer
        w51 = w51 + step3*der_sig(t3)*e*x31
        w52 = w52 + step3*der_sig(t3)*e*x32
        b3 = b3 + step3*der_sig(t3)*e*x33
        #hidden layer 2
        w31 = w31 + step2*der_sig(t21)*(e*der_sig(t3)*w51)*x21
        w32 = w32 + step2*der_sig(t22)*(e*der_sig(t3)*w52)*x21
        w41 = w41 + step2*der_sig(t21)*(e*der_sig(t3)*w51)*x22
        w42 = w42 + step2*der_sig(t22)*(e*der_sig(t3)*w52)*x22
        b21 = b21 + step2*der_sig(t21)*(e*der_sig(t3)*b3)*x23
        b22 = b22 + step2*der_sig(t22)*(e*der_sig(t3)*b3)*x23
        #hidden layer 1
        w11 = w11 + step1*der_sig(t11)*(der_sig(t21)*w31*e*der_sig(t3)*w51+der_sig(t22)*w32*e*der_sig(t3)*w52)*x11
        w12 = w12 + step1*der_sig(t12)*(der_sig(t21)*w41*e*der_sig(t3)*w51+der_sig(t22)*w42*e*der_sig(t3)*w52)*x11
        w21 = w21 + step1*der_sig(t11)*(der_sig(t21)*w31*e*der_sig(t3)*w51+der_sig(t22)*w32*e*der_sig(t3)*w52)*x12
        w22 = w22 + step1*der_sig(t12)*(der_sig(t21)*w41*e*der_sig(t3)*w51+der_sig(t22)*w42*e*der_sig(t3)*w52)*x12
        b11 = b11 + step1*der_sig(t11)*(der_sig(t21)*w31*e*der_sig(t3)*w51+der_sig(t22)*w32*e*der_sig(t3)*w52)*x13
        b12 = b12 + step1*der_sig(t12)*(der_sig(t21)*w41*e*der_sig(t3)*w51+der_sig(t22)*w42*e*der_sig(t3)*w52)*x13
        error[i] = e

t = np.matrix(np.arange(-2,2,0.1))
z11 = (-w11/w21)*t + (-b11/w21)
z12 = (-w12/w22)*t + (-b12/w22)
z21 = (-w31/w41)*t + (-b21/w41)
z22 = (-w32/w42)*t + (-b22/w42)

print((-w11/w21))
print((-b11/w21))

fig = plt.figure(figsize=(20,10))
p1 = fig.add_subplot(*[2,1,1])
p1.scatter(training[0:100,0],training[0:100,1],c="b")
p1.scatter(training[100:200,0],training[100:200,1],c="b")
p1.scatter(training[200:300,0],training[200:300,1],c="r")
p1.scatter(training[300:400,0],training[300:400,1],c="r")
p1.plot(t,z11)
p1.plot(t,z12)
ymin, ymax = -2, 2
p1.set_ylim([ymin,ymax])

p2 = fig.add_subplot(*[2,1,2])
p2.scatter(x[0:100],v[0:100],c="b")
p2.scatter(x[100:200],v[100:200],c="b")
p2.scatter(x[200:300],v[200:300],c="r")
p2.scatter(x[300:400],v[300:400],c="r")
p2.plot(t,z21)
p2.plot(t,z22)
ymin, ymax = -2, 2
p2.set_ylim([ymin,ymax])

print('finish')