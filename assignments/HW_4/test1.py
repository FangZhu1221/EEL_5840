# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:59:10 2017

@author: JSZJZ
"""

import numpy as np
import matplotlib.pyplot as plt
import math 

fig = plt.figure(figsize=(10,10))

# Maximum Likelihood Solution 
# generate data
upper = 1 #known
lower = 0 #known
num = 100 #known
u_prior = 0.5
sigma_prior = 0.1 #known
sigma_likelihood = 100 #known
prior = np.zeros(num)
posterior = np.zeros(num)

d = np.arange(lower,upper,upper/num)
data = np.matrix(d)
# compute u_ml and sigma_ml
u_ml = sum(d)/num # u_ml
I = np.ones([1,num])
u_l = np.matrix(u_ml * I)
sigma_ml = np.asscalar((data - u_l)@(data - u_l).T/num) # sigma_ml

#compute u_map and sigma_map
u_map = (u_prior*(sigma_likelihood**2))/(num*sigma_prior**2+sigma_likelihood**2)+(u_ml*num*(sigma_prior**2))/(num*sigma_prior**2+sigma_likelihood**2) # u_map
sigma_map = 1/((1/sigma_prior+num/sigma_likelihood))
print(sigma_map)
print(sigma_ml)

for i in range(num):
    prior[i] = (1/(math.sqrt(2*math.pi*sigma_prior)))*math.exp((-0.5/sigma_ml)*(d[i]-u_prior)*(d[i]-u_prior))
    posterior[i] = (1/(math.sqrt(2*math.pi*sigma_map)))*math.exp((-0.5/sigma_map)*(d[i]-u_map)*(d[i]-u_map))

p1 = fig.add_subplot(*[1,1,1])
p1.plot(d,prior, 'r')
p1.plot(d,posterior,'b')
