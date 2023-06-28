#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 13:26:51 2019

@author: atilman
"""

import numpy as np
from scipy.integrate import solve_ivp
from multiprocessing import Pool
# import pylab as plt
from scipy.fftpack import fft
# import matplotlib.animation as animation

def invasion(eps1,eps2,r,C,DT_L1,dt_H0,dt_L0,DT_H1,zLm0,n0,tEnd,tFinal,sims,invFrac,Linv):
    """
    eps1: reltave timescale of the environment
    eps2: relative timescale of siwtching from forecaster to myopic and visa versa
    r: discount rate
    C: cost of forecasting
    Dt_L1: incentive to switch to HIGH impact (capital Dt), given other follow strategy L and the environment is rich (1) 
    DT_H1: incentive to switch to HIGH impact, given others follow strategy H and the environment is rich
    dt_L0: incentive to switch to LOW impact, given others follow strategy H and the environment is poor
    dt_H0: incentive to switch to LOW impact, given others follow strategy L and the environment is poor
    zLm0: initial frequency of Low impact strategy (myopic only setting)
    n0: init environmental state
    tEnd: end of myopic only simulaton
    tFinal: end of whole population simulation
    sims: number of invasion points
    invFrac: frequency of invaders
    Linv: frequency of L among invaders (if set to a string, then copy resident pop upon invasion)
    """
    args = (eps1,eps2,r,C,dt_H0,dt_L0,DT_H1,DT_L1)
    if zLm0>1:
        print("these are non-sensical initial conditions!")
    else:
        zHm0 = 1-zLm0
        W0 = [0, 0, zLm0, zHm0, n0]
        fun = lambda t,y: altSys(y,t,*args)
        sol = solve_ivp(fun,[0,tEnd],W0,rtol=10**-13,atol=10**-14)
        Q0 = [zLm0, zHm0, 0, 0, n0]
        solFore = solve_ivp(fun,[0,80],Q0,rtol=10**-13,atol=10**-14)
        #zLf = sol.y[0]
        #zHf = sol.y[1]
        zLm = sol.y[2]
        zHm = sol.y[3]
        n = sol.y[4]
        t = sol.t
        (cycletimePre,peaksPre,oneCycle)=fourier(t,n)
        "INVASION SIMULATIONS"
        tInvStart = t[np.argmax(t>t[-1]-oneCycle)]
        #tInv = np.linspace(tInvStart,tEnd,num=sims)
        tInv = np.arange(tInvStart,tEnd,(tEnd-tInvStart)/sims)
        "INVASION STRUCTURE"
        Invfrac = invFrac
        if Linv<1 and Linv>0:
            Lfrac = np.full(sims,Linv)
        else:
            Lfrac = np.interp(tInv,t,zLm)
        zLmInv = (1-Invfrac)*np.interp(tInv,t,zLm)
        zHmInv = (1-Invfrac)*np.interp(tInv,t,zHm)
        zLfInv = Invfrac*Lfrac
        zHfInv = Invfrac*(1-Lfrac)
        nInv = np.interp(tInv,t,n)
        A = np.empty((sims,11),dtype=object)
        for i in range(sims):
            W0Inv = [zLfInv[i],zHfInv[i],zLmInv[i],zHmInv[i],nInv[i]]
            A[i] = [eps1,eps2,r,C,dt_H0,dt_L0,DT_H1,DT_L1,W0Inv,tInv[i],tFinal]
        #if __name__=='__main__':
        with Pool(7) as pool:        
            simOut = pool.map(altSysRun,A)
        output = [simOut,sol,tEnd,tFinal,sims,eps1,eps2,r,C,DT_L1,dt_H0,dt_L0,DT_H1,zLm0,n0,invFrac,Linv,solFore]        
    return(output)

def invasionVary(eps1,eps2,rVals,C,DT_L1,dt_H0,dt_L0,DT_H1,zLm0,n0,tEnd,tFinal,sims,invFrac,Linv):
    """
    eps1: reltave timescale of the environment
    eps2: relative timescale of siwtching from forecaster to myopic and visa versa
    rVals: discount rates tested
    C: cost of forecasting
    Dt_L1: incentive to switch to HIGH impact (capital Dt), given other follow strategy L and the environment is rich (1) 
    DT_H1: incentive to switch to HIGH impact, given others follow strategy H and the environment is rich
    dt_L0: incentive to switch to LOW impact, given others follow strategy H and the environment is poor
    dt_H0: incentive to switch to LOW impact, given others follow strategy L and the environment is poor
    zLm0: initial frequency of Low impact strategy (myopic only setting)
    n0: init environmental state
    tEnd: end of myopic only simulaton
    tFinal: end of whole population simulation
    sims: number of invasion points
    invFrac: frequency of invaders
    Linv: frequency of L among invaders (if set to a string, then copy resident pop upon invasion)
    """
    args = (eps1,eps2,rVals[0],C,dt_H0,dt_L0,DT_H1,DT_L1)
    if zLm0>1:
        print("these are non-sensical initial conditions!")
    else:
        zHm0 = 1-zLm0
        W0 = [0, 0, zLm0, zHm0, n0]
        fun = lambda t,y: altSys(y,t,*args)
        sol = solve_ivp(fun,[0,tEnd],W0,rtol=10**-13,atol=10**-14)
        #zLf = sol.y[0]
        #zHf = sol.y[1]
        zLm = sol.y[2]
        zHm = sol.y[3]
        n = sol.y[4]
        t = sol.t
        (cycletimePre,peaksPre,oneCycle)=fourier(t,n)
        "INVASION SIMULATIONS"
        tInvStart = t[np.argmax(t>t[-1]-oneCycle)]
        #tInv = np.linspace(tInvStart,tEnd,num=sims)
        tInv = np.arange(tInvStart,tEnd,(tEnd-tInvStart)/sims)
        "INVASION STRUCTURE"
        Invfrac = invFrac
        if Linv<1 and Linv>0:
            Lfrac = np.full(sims,Linv)
        else:
            Lfrac = np.interp(tInv,t,zLm)
        zLmInv = (1-Invfrac)*np.interp(tInv,t,zLm)
        zHmInv = (1-Invfrac)*np.interp(tInv,t,zHm)
        zLfInv = Invfrac*Lfrac
        zHfInv = Invfrac*(1-Lfrac)
        nInv = np.interp(tInv,t,n)
        A = np.empty((sims,11),dtype=object)
        xfMal = np.empty((len(rVals),sims),dtype=object)
        for k in range(len(rVals)):
            for i in range(sims):
                W0Inv = [zLfInv[i],zHfInv[i],zLmInv[i],zHmInv[i],nInv[i]]
                A[i] = [eps1,eps2,rVals[k],C,dt_H0,dt_L0,DT_H1,DT_L1,W0Inv,tInv[i],tFinal]
            #if __name__=='__main__':
            with Pool(7) as pool:        
                simOut = pool.map(altSysRun,A)
            for j in range(sims):
                zLfPost = simOut[j].y[0]
                zHfPost = simOut[j].y[1]
                xfPost = zLfPost + zHfPost
                xfMal[k,j] = xfPost[-1]
            output = [xfMal,tEnd,tFinal,sims,eps1,eps2,rVals,C,DT_L1,dt_H0,dt_L0,DT_H1,zLm0,n0,invFrac,Linv]        
    return(output)
    
    
def altSys(W,t0,eps1,eps2,r,C,d00,d10,d01,d11):
    """
    W[0]: Frequency of forecasters with strategy L: `zLf'
    W[1]: Frequency of forecasters witehr strategy H: `zHf'
    W[2]: Frequency of myopic agents with strategy L: `zLm'
    w[3]: Frequency of myopic agents with strategy H: `zHm'
    W[4]: Environmental state: `n'
    FOR COMPETITION EXCEPT AMONG FORECASTERS
    """
    "Defs"
    zLf = W[0] 
    zHf = W[1] 
    zLm = W[2] 
    zHm = W[3]
    n = W[4]
    zL = zLf+zLm
    "Selection gradients"
    g = (1-n)*(d10*zL + d00*(1-zL)) - n*(d11*zL + d01*(1-zL))
    dgdn = -(d10*zL + d00*(1-zL)) - (d11*zL + d01*(1-zL))
    fLsel = g + eps1*(zL-n)*dgdn/r 
    "dynamical equations"
    dzLf =   zLf*zHf*fLsel - eps2*zLf*zLm*C + eps2*zLf*zHm*(g-C)
    dzHf = - zLf*zHf*fLsel - eps2*zHf*zLm*(g+C) - eps2*zHf*zHm*C 
    dzLm =   zLm*zHm*g + eps2*zLm*zLf*C + eps2*zLm*zHf*(g+C)
    dzHm = - zLm*zHm*g + eps2*zHm*zLf*(C-g) + eps2*zHm*zHf*C
    dn = eps1*(zL-n)
    return([dzLf, dzHf, dzLm, dzHm, dn])

    
def altSysRun(A):
    eps1=A[0]
    eps2=A[1]
    r=A[2]
    C=A[3]
    d00=A[4]
    d10=A[5]
    d01=A[6]
    d11=A[7]
    W0=A[8]
    tStart=A[9]
    tEnd=A[10]
    args=(eps1,eps2,r,C,d00,d10,d01,d11)
    fun = lambda t,y: altSys(y,t,*args)
    sol = solve_ivp(fun,[tStart,tEnd],W0,rtol=10**-13,atol=10**-14)
    return(sol)



def fourier(tData,yData):
    'tData: time points in simulation'
    'yData: corresponding observed values in time series'
    tNew = np.linspace(tData[0],tData[-1],num=len(tData))
    yNew = np.interp(tNew,tData,yData)
    yFourier = np.abs(fft(yNew)[1:])
    stepsize = tNew[1]-tNew[0]
    freq = np.linspace(0,1/stepsize,num=len(tData))
    cycleTime = 1/freq[1:]
    cycle = cycleTime[np.argmax(yFourier)]
    return(cycleTime,yFourier,cycle)




"AVERAGER FUNCTION"
def avg(x,y):
    yTotal=0
    for i in range(len(x)-1):
        yTotal=yTotal+y[i]*(x[i+1]-x[i])
    yMean = yTotal/(x[-1]-x[0])
    return(yMean)

    
def sys2(W,t0,eps1,eps2,r,C,d00,d10,d01,d11):
    """
    W[0]: Frequency of strategy 1 among myopic harvesters: `y1m'
    W[1]: Frequency of strategy 1 among forecasting harvesters: `y1f'
    W[2]: Frequency of forcasting agents: `xf'
    W[3]: Environmental state: `n'
  
    hierarchical-type switching is based on insta payoff
    
    """
    "Defs"
    y1m = W[0] 
    y1f = W[1] 
    xf = W[2] 
    n = W[3] 
    x1 = y1f*xf+y1m*(1-xf)
    
    "Gradient of selection without forecasting and partial deriv"
    g = (1-n)*(d10*x1 + d00*(1-x1)) - n*(d11*x1 + d01*(1-x1))
    dgdn = -(d10*x1 + d00*(1-x1)) - (d11*x1 + d01*(1-x1))
    
    "dynamical equations"
    dy1m = y1m*(1-y1m)*g
    dy1f = y1f*(1-y1f)*(g + eps1*(x1-n) * dgdn/r) #CONTS forecasting with discounting
    dxf = eps2*xf*(1-xf)*((y1f-y1m)*g-C)
    dn = eps1*(x1-n)
    
    return([dy1m, dy1f, dxf, dn])


    
def sys2Run(A):
    eps1=A[0]
    eps2=A[1]
    r=A[2]
    C=A[3]
    d00=A[4]
    d10=A[5]
    d01=A[6]
    d11=A[7]
    W0=A[8]
    tEnd=A[9]
    args=(eps1,eps2,r,C,d00,d10,d01,d11)
    fun = lambda t,y: sys2(y,t,*args)
    sol = solve_ivp(fun,[0,tEnd],W0,rtol=10**-13,atol=10**-14)
    return(sol)
  