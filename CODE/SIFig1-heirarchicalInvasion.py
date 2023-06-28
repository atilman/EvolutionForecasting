#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 13:41:51 2018

@author: atilman
This simulation lets myopics approach a long term outcome, then tests invasion by forecasters

"""

#from scipy.integrate import solve_ivp
import pylab as plt
import random as rand
#import numpy as np
from scipy.integrate import solve_ivp
import SimulationBackbone as sb
"Do you want to save this simulation figure???"

save = 'ys'

"PARAMS"
eps1 = 3/10 #env feedback speed (relative to strategy switching speed)
eps2 = 1 #speed of switching types (frecaster vs myopic) (relative to strategies)
r=15/100 #discount rate: lower number care more aobut future
C = 1/100 #cost of forecasting
d11 = 2 #incentive parameters
d00 = 1
d10 = 3
d01 = 4
tStart=0
tInv=[105,100,95]
tEnd = [800,800,2000]
args = (eps1,eps2,r,C,d00,d10,d01,d11)

"COLORS"
forecasterColor = "darkorchid"               # HTML name
myopicColor = "darkcyan"


mFit = (0.0953921568627451, 0.449705882352941, 0.4497058823529411)
fFit = (0.593895594601477, 0.21415686274509815, 0.7819215686274509)

for i in range(3):
    "INITIAL CONDITIONS"
    y1m0 = .4
    y1f0 = 0
    xf0 = 0
    n0 = .06
    
    "PRE-Invasion simulation"
    W0 = [y1m0,y1f0,xf0,n0]
    fun=lambda t,y: sb.sys2(y,t,*args)
    sol = solve_ivp(fun,[0,tInv[i]],W0,rtol=10**-13,atol=10**-14)
    y1mPre=sol.y[0]
    nPre=sol.y[3]
    tPre=sol.t
    
    
    "Post-Invasion simulation"
    popSize=[100,100,100]
    y1m01 = y1mPre[-1]
    y1f01 = y1mPre[-1] #same frac as myopics
    xf01 = 1/popSize[i]
    n01 = nPre[-1]
    W0 = [y1m01,y1f01,xf01,n01]
    fun=lambda t,y: sb.sys2(y,t,*args)
    sol = solve_ivp(fun,[tInv[i],tEnd[i]],W0,rtol=10**-13,atol=10**-14)
    y1m=sol.y[0]
    y1f=sol.y[1]
    xf=sol.y[2]
    n=sol.y[3]
    t=sol.t
    y1=xf*y1f+(1-xf)*y1m
    
    #%%
    pltStart=tStart
    pltInv=tInv[i]
    pltEnd=tEnd[i]
    phaseFrac=.25
    invFrac=.5
    
    preStart = int(len(tPre)-phaseFrac*len(tPre))
    preEnd = len(tPre)
    invStart = 0
    invEnd = int(invFrac*len(t))
    postStart = int(len(t)-phaseFrac*len(t))
    postEnd = len(t)
    
    
    #%%
    "PHASE DIAGRAMS"
    fig2 = plt.figure(figsize=(10,5))
    ax30 = fig2.add_subplot(1,2,1)
    ax31= fig2.add_subplot(1,2,2)
    
    "PRE invasion and just post invasion"
    ax30.plot(y1mPre[preStart:preEnd], nPre[preStart:preEnd],'k-',alpha=.3,linewidth=5,label='Myopic limit cycle')
    if i != 2:
        ax30.plot(y1[postStart:postEnd], n[postStart:postEnd],'k-',alpha=.5,linewidth=5, label='Population-level post invasion limit')
    ax30.plot(y1[invStart:invEnd], n[invStart:invEnd],'k--',linewidth=2,alpha=.7, label='Population-level invasion')
    
    
    'PRE invasion vs POST invasion'
    ax31.plot(y1mPre[preStart:preEnd], nPre[preStart:preEnd],'k-',alpha=.3,linewidth=5)
    if i != 2:
        ax31.plot(y1[postStart:postEnd], n[postStart:postEnd],'k-',alpha=.5,linewidth=5)
        ax31.plot(y1f[postStart:postEnd], n[postStart:postEnd],color = fFit,alpha=.5,linewidth=5, label='Post invasion limit, forecaster')
        ax31.plot(y1m[postStart:postEnd], n[postStart:postEnd],color = mFit,alpha=.5,linewidth=5, label='Post invasion limit, myopic')
    ax31.plot(y1f[invStart:invEnd], n[invStart:invEnd],color = fFit,linewidth=2,alpha=.7,linestyle='dashed', label='Forecasters invasion')
    ax31.plot(y1m[invStart:invEnd], n[invStart:invEnd],color = mFit,linewidth=2,alpha=.7,linestyle='dashed', label='Myopic invasion')
    
    "InvasionPoint"
    ax30.plot([y1[0]],[n01],'ko',markersize=12)
    ax31.plot([y1[0]],[n01],'ko',markersize=12)
    
    "Plot settings"
    ax30.set_xlim((0,1))
    ax30.set_ylim((0,1))
    ax30.set_xlabel("L-strategy fraction")
    ax30.set_ylabel("Environmental state")
    ax30.grid()
    ax30.legend(loc='best')
    ax31.set_xlim((0,1))
    ax31.set_ylim((0,1))
    ax31.set_xlabel("L-strategy fraction")
    ax31.set_ylabel("Environmental state")
    ax31.grid()
    ax31.legend(loc='best')
    
    
    # paramAnnote='eps1='+str(eps1)+', eps2='+str(eps2)+', r='+str(r)+', C='+str(C)+', d11='+str(d11)+', d00='+str(d00)+', d10='+str(d10)+', d01='+str(d01)+', tStart='+str(tStart)+', tInv='+str(tInv[i])+', tEnd='+str(tEnd[i])+', y1m0='+str(y1m0)+', n0='+str(n0)+', y1f01='+str(y1f01)[0:5]+', xf01='+str(xf01)[0:5]+', y1m01='+str(y1m01)[0:5]+', n01='+str(n01)[0:5]+', pltEnd='+str(pltEnd)+', phaseFrac='+str(phaseFrac)+', invFrac='+str(invFrac)         
    #pars1 = ax2.annotate(paramAnnote, (0,0), (-20, -40), xycoords='axes fraction', textcoords='offset points', va='top',fontsize=6)
    
    #pars2 = ax30.annotate(paramAnnote, (0,0), (-20, -40), xycoords='axes fraction', textcoords='offset points', va='top',fontsize=6)
    
    if save == 'yes':
        tag=['a','b','c']
        fig2.savefig('../FIGS/'+'SIFig1'+tag[i]+'.png',dpi=300, bbox_inches='tight')
    


































