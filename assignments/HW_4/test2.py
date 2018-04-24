# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 16:52:23 2017

@author: JSZJZ
"""

import numpy as np
import matplotlib.pyplot as plt
import math 

fig = plt.figure(figsize=(10,20))

mean = 1
sigma = 10
u_prior = 1
sigma_prior = 0.1
step = 50
group = 200

box = np.zeros(step*group)
U_ML = np.zeros(group)
U_MAP = np.zeros(group)
U = mean*np.ones(group)
S_ML = np.zeros(group)
S_MAP = np.zeros(group)
S = sigma*np.ones(group)
U_P = np.ones(group)*u_prior
S_P = np.ones(group)*sigma_prior

for it in range(group):
    for i in range(step):
        box[i+it*step] = np.random.normal(mean,sigma,1)
    n = (it+1)*step
    data = np.zeros(n)
    data = box[0:(it+1)*step]
    # ML
    u_ml = sum(data)/n
    I = np.ones([1,n])
    u_l = np.matrix(u_ml * I)
    sigma_ml = math.sqrt(np.asscalar((data - u_l)@(data - u_l).T/n))
    # MAP
    u_map = (u_prior*(sigma**2))/(n*sigma_prior**2+sigma**2)+(u_ml*n*(sigma_prior**2))/(n*sigma_prior**2+sigma**2) # u_map
    sigma_map = math.sqrt(1/((1/(sigma_prior**2)+n/(sigma**2))))
    print('1. The mean of the data: '+str(mean))
    print('2. The sigma of the data: '+str(sigma))
    print('3. The mean of the prior: '+str(u_prior))
    print('4. The sigma of the prior: '+str(sigma_prior))
    print('5. The mean of the ML: '+str(u_ml))
    print('6. The sigma of the ML: '+str(sigma_ml))
    print('7. The mean of the MAP: '+str(u_map))
    print('8. The sigma of the MAP: '+str(sigma_map))
    print('9. Error between the mean of MAP and ML: '+str(abs(u_map-u_ml)))
    print(' ')
    print('***************************************************************')
    print(' ')
    # update
    u_prior = u_map
    sigma_prior = sigma_map
    U_ML[it] = u_ml
    U_MAP[it] = u_map
    S_ML[it] = sigma_ml
    S_MAP[it] = sigma_map
    

p1 = fig.add_subplot(*[2,1,1])
t = np.arange(step,step+step*group,step)
p1.plot(t,U_ML, 'g')
p1.plot(t,U_MAP,'b')
p1.plot(t,U,'r')
#p1.plot(t,U_P,'y')

p2 = fig.add_subplot(*[2,1,2])
#p2.plot(t,S_ML, 'g')
p2.plot(t,S_MAP,'b')
p2.plot(t,S,'r')
p2.plot(t,S_P,'y')