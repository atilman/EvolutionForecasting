#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 16:39:44 2018

@author: atilman
"""


import numpy as np
import scipy.integrate as sp_int
import pylab as plt

def dndx(W,t0,epsilon,d00,d10,d01,d11):
    """
    W[0]: Enviornmental state measure `n'
    W[1]: Fraction of high effort harvesters `x'
    """
    
    "Gradient of selection"
    g = (1-W[0])*(d10*W[1]+d00*(1-W[1]))-W[0]*(d11*W[1]+d01*(1-W[1]))
        
    "dynamical equations"
    dx = W[1]*(1-W[1])*(g)
    dn = epsilon*(W[1]-W[0])
  
    return([dn, dx])
  
RUN=16;  
epsilon=25/100;
draft=True 


T1 = 6
R1 = 4
P1 = 4
S1 = 3

R0 = 6
T0 = 2
S0 = 2
P0 = 17/8


DT_L1 = T1 - R1;

DT_H1 = P1 - S1;
dt_L0 = R0 - T0;

dt_H0 = S0 - P0;

d11 = DT_L1
d01 = DT_H1

d00 = dt_H0
d10 = dt_L0

numSims = 8;
tEnd=200
steps=5000

params = (epsilon,d00,d10,d01,d11)
args = (epsilon,d00,d10,d01,d11)
paramAnnote='epsilon='+str(epsilon)+', d11='+str(d11)+', d00='+str(d00)+', d10='+str(d10)+', d01='+str(d01)+', tEnd='+str(tEnd)+', numSims='+str(numSims)           
opacity=.2
initVarx=0.02
initVarn=0.05
initx=np.linspace(.1,.9,numSims)
initn=np.linspace(.1,.9,numSims)
W0 = [initn, initx]
tRange=np.linspace(0,tEnd,steps)
results=np.zeros((numSims,numSims,len(tRange),2))
result=np.empty((numSims,numSims))
fun=lambda t,y: dndx(y,t,*args)
solNX=[]
solT=[]
for i in range(numSims):
    for j in range(numSims):
        W0 = [initn[i]+np.random.normal(0,initVarn,1)[0], initx[j]+np.random.normal(0,initVarx,1)[0]]
        results[i,j,:,:] = sp_int.odeint(dndx,y0=W0,t=tRange,args=params,rtol=10**-13,atol=10**-14)

        
fig = plt.figure(figsize=(5,5))

ax2 = fig.add_subplot(1,1,1)



frac=np.linspace(0,1,1000)
xNullcline=((d00)*(1-frac)+(d10)*frac)/((d01+d00)*(1-frac)+(d11+d10)*frac)
xNullcline[xNullcline>1]=np.nan
xNullcline[xNullcline<0]=np.nan


for i in range(numSims):
    for j in range(numSims):
        if (d11<0 and d00<0) or (d11<0 and d00>0 and d01-d10>2*(-d11*d00)**.5) or (d11>0 and d00<0 and d01-d10<-2*(-d11*d00)**.5):
            if np.absolute(1-results[i,j,-1,1])<.01 or np.absolute(results[i,j,-1,1])<.01:
                hue = 1
            else:
                hue = .4
            ax2.plot(results[i,j,:,1], results[i,j,:,0], color=(0, hue, 1-hue),alpha=opacity,rasterized=draft)
        else:
            ax2.plot(results[i,j,:,1], results[i,j,:,0], color="black",alpha=opacity,rasterized=draft)
            
ax2.plot([0,1],[0,1],color="blue") 
ax2.plot(frac,xNullcline,color="r") 
ax2.plot([0.004,0.004],[0,1],color="r") 
ax2.plot([.996,.996],[0,1],color="r") 
ax2.set_ylim((0,1))
ax2.set_xlim((0,1))
ax2.set_xlabel("Low-impact fraction")
ax2.set_ylabel("Environmental state")  
ax2.set_title("Phase plane")

eCrit = (((4*d11*d00+(d01-d10)**2)**(1/2)-d10-d01)*((4*d11*d00+(d01-d10)**2)**(1/2)-2*d11+d01-d10)*((4*d11*d00+(d01-d10)**2)**(1/2)-2*d00-d01+d10))/(8*(d11+d10-d01-d00)**2)
print(eCrit)

fig.savefig('../FIGS/SIFig7.png',dpi=500, bbox_inches='tight')











































