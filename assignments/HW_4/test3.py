# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 13:15:45 2017

@author: JSZJZ
"""

import numpy as np
import matplotlib.pyplot as plt
import math 

fig = plt.figure(figsize=(10,20))

mean = 1
sigma = 0.1
u_prior = 1
sigma_prior = 0.1
group = 1000
step = 3

box = np.zeros(step*group)
U_ML = np.zeros(group)
U_MAP = np.zeros([step,group])
U = mean*np.ones(group)
S_ML = np.zeros(group)
S_MAP = np.zeros([step,group])
S = sigma*np.ones(group)
U_P = np.ones([step,group])
S_P = np.ones([step,group])

for it in range(group):  
    box[it] = np.random.normal(mean,sigma,1)
    n = (it+1)
    data = np.zeros(n)
    data = box[0:(it+1)]
    # ML
    u_ml = sum(data)/n
    I = np.ones([1,n])
    u_l = np.matrix(u_ml * I)
    sigma_ml = math.sqrt(np.asscalar((data - u_l)@(data - u_l).T/n))
    U_ML[it] = u_ml
    S_ML[it] = sigma_ml
    for i in range([0.1,1,10]):
        # MAP
        u_map = (i*(sigma**2))/(n*sigma_prior**2+sigma**2)+(u_ml*n*(sigma_prior**2))/(n*sigma_prior**2+sigma**2) # u_map
        sigma_map = math.sqrt(1/((1/(sigma_prior**2)+n/(sigma**2))))
        if it == group-1:
            print(' ')
            print('1. The mean of the data: '+str(mean))
            print('2. The sigma of the data: '+str(sigma))
            print('3. The mean of the prior: '+str(i))
            print('4. The sigma of the prior: '+str(sigma_prior))
            print('5. The mean of the ML: '+str(u_ml))
            print('6. The sigma of the ML: '+str(sigma_ml))
            print('7. The mean of the MAP: '+str(u_map))
            print('8. The sigma of the MAP: '+str(sigma_map))
            print('9. Error between the mean of MAP and ML: '+str(abs(u_map-u_ml)))
            print(' ')
            # update
        u_prior = u_map
        sigma_prior = sigma_map
        U_MAP[it] = u_map
        S_MAP[it] = sigma_map
        U_P[]
        
    



p2 = fig.add_subplot(*[2,1,2])
#p2.plot(t,S_ML, 'g')
p2.plot(t,S_MAP,'b')
p2.plot(t,S,'r')
p2.plot(t,S_P,'y')