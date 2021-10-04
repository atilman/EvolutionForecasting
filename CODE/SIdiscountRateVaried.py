#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:41:58 2020

@author: atilman
a script that test the sucess of invasions against the discount rate
"""

import pylab as plt
#from scipy.integrate import solve_ivp
import numpy as np
#from scipy.fftpack import fft
#from multiprocessing import Pool
import SimulationBackbone as sb
import random as rand
import time

"GENERAL PARAMS"
eps1 = 3/10;
eps2 = 1/10;
rVals=np.arange(1/100,35/100,1/100);
#rVals = np.array([1/10000,1/1000,1/100,1/10,2/10,3/10,4/10,5/10])
C = 5/1000;
DT_L1 = 2;
dt_H0 = 1;
dt_L0 = 3;
DT_H1 = 4;
sims = 56
popSize=100
invFrac=1/popSize
Linv=990/100

"SAVE???"
save='yes'
tag=str(rand.random())[2:5]

"SIM"
zLm0 = 3/10
n0 = 4/10
tEnd = 300
tFinal = 10000
simStart=time.time()
D = sb.invasionVary(eps1,eps2,rVals,C,DT_L1,dt_H0,dt_L0,DT_H1,zLm0,n0,tEnd,tFinal,sims,invFrac,Linv)
simEnd=time.time()
print(str((simEnd-simStart)/60)[0:6]+'minutes')
if save == 'yes':
    np.save('../FIGS/InvasionSuccessRate'+tag+'.npy',D)
#%%
"""
PLOTS
"""
"Set tag manually if desired"
tag='894'
save='yes'
Data = np.load('../FIGS/InvasionSuccessRate'+tag+'.npy',allow_pickle=True)
xfMal = Data[0]
rVals = Data[6]
InvProb = np.empty((len(rVals)),dtype=object)
forecasterFreq = np.empty((len(rVals)),dtype=object)
freqWhenFail = np.empty((len(rVals)),dtype=object)
for i in range(len(rVals)):
    count = 0
    freqAvg = 0
    failAvg = 0
    for j in range(sims):
        if xfMal[i,j] > invFrac:
            count = count + 1
            freqAvg = freqAvg + xfMal[i,j]
        else:
            failAvg = failAvg + xfMal[i,j]
            
    InvProb[i] = count / sims
    if count !=0:
        forecasterFreq[i] = freqAvg / count
    if count != sims:
        freqWhenFail[i] = failAvg / (sims-count)
fig1 = plt.figure(figsize=(8,4))
ax1 = fig1.add_subplot(111)
#ax1.set_xscale('log')
ax1.plot(rVals,forecasterFreq,'*', label = 'Long-run fraction of forecasters, given invasion')
#ax1.plot(rVals,freqWhenFail,'*', label = 'Forecaster long-run frequency when invasion fails')
ax1.plot(rVals,InvProb,'*',label='Invasion probability')
ax1.set_title("Invasion outcomes")
ax1.set_xlabel('Discount rate, r')
ax1.legend(loc='best')
ax1.set_ylim((0,1))

if save == 'yes':
    fig1.savefig('../FIGS/SIFig3.png',dpi=300)

