#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:03:18 2020

@author: atilman
INV SIMULATION
"""
import pylab as plt
#from scipy.integrate import solve_ivp
import numpy as np
#from scipy.fftpack import fft
#from multiprocessing import Pool
import SimulationBackbone as sb
import random as rand
#import time

"GENERAL PARAMS"
eps1 = 3/10;
eps2 = 1/10;
r=5/100;
C = 5/1000;
DT_L1 = 2;
dt_H0 = 1;
dt_L0 = 3;
DT_H1 = 4;
sims = 28                                                                                                                                                                                                                                                                                
popSize=550
invFrac=1/popSize
Linv=990/100

"NAME TAG"
tag=str(rand.random())[2:5]
tag = "100"
"SIMULATION"
zLm0 = 3/10
n0 = 9/10
tEnd = 800
tFinal = 15000
dataOut = sb.invasion(eps1,eps2,r,C,DT_L1,dt_H0,dt_L0,DT_H1,zLm0,n0,tEnd,tFinal,sims,invFrac,Linv)
np.save('../FIGS/mainTextFig3_'+tag+'.npy',dataOut)
#%%
"""
PLOTTING
"""

"SAVE???"
save = 'ys'
final = 'yes'

"MANUALLY SET TAG"
tag = '100'
"Read in data"
output = np.load('../FIGS/mainTextFig3_'+tag+'.npy',allow_pickle=True)

"Set Vars"
simOut = output[0]
sol = output[1]
eps1 = output[5]
eps2 = output[6]
r = output[7]
C = output[8]
DT_L1 = output[9]
dt_H0 = output[10]
dt_L0 = output[11]
DT_H1 = output[12]
zLm0 = output[13]
n0 = output[14]
tEnd = output[2]
tFinal = output[3]
sims = output[4]
invFrac = output[15]
Linv = output[16]
solFore = output[17]

zLm = sol.y[2]
zHm = sol.y[3]
n = sol.y[4]
t = sol.t
(cycletimePre,peaksPre,oneCycle)=sb.fourier(t,n)
Shown = 5
plotStart = np.argmax(t>t[-1]-Shown*oneCycle)
InvStart = np.argmax(t>t[-1]-oneCycle)
tInvStart = t[InvStart]

tPlotEnd = np.min((10000,tFinal))
#XFc = 'g-'
# ENVc = 'y-'
# LMc = 'c-'
# HMc = 'm-'
# LFc = 'darkorchid'
# HFc = 'r-'

"PlotColors"
import seaborn as sns
forecasterColor = "darkorchid"               # HTML name
myopicColor = "darkcyan"
freqColor = "dimgrey"
ENVc = 'y'
LMc = sns.set_hls_values(color = myopicColor, h = None, l = None, s = .25)
HMc = sns.set_hls_values(color = myopicColor, h = None, l = None, s = 1)
LFc = sns.set_hls_values(color = forecasterColor, h = None, l = None, s = .15)
HFc = sns.set_hls_values(color = forecasterColor, h = None, l = None, s = 1)
mFit = sns.set_hls_values(color = myopicColor, h = None, l = None, s = .65)
fFit = sns.set_hls_values(color = forecasterColor, h = None, l = None, s = .57)
sInv = sns.set_hls_values(color = freqColor, h = None, l = None, s = 1)
uInv = sns.set_hls_values(color = freqColor, h = None, l = None, s = .2)
opac = .5
"PlotWidths"
envWidth = 2
stratWidth = 2

fig2 = plt.figure(figsize=(10,5))
ax2 = fig2.add_subplot(122)
ax2.set_yscale('log')
ax2.set_xlabel('Time')
ax2.set_xlim((tInvStart,tPlotEnd))
ax2.set_ylim((.00001,1))
ax2.set_ylabel('Log-scale fraction of forecasters')
ax2.set_title('Fraction of forecasting types')

inset = plt.axes([.56, 0.65, .1, .2])
inset.set_xlim((tInvStart-.1*oneCycle,tEnd+oneCycle))
inset.set_ylim((.68*invFrac,1.05*invFrac))


#inset.set_yscale('log')
# inset.plot(t,n,ENVc,linewidth=2,alpha=1,label='Env state')
# inset.plot(t,zLm,LMc,linewidth=1,alpha=.5,label='freq of L-strategy mypoic')
# inset.plot(t,zHm,HMc,linewidth=1,alpha=.5,label='freq of H-strategy mypoic')
# inset.plot(t,zLf,LFc,linewidth=1,alpha=.5,label='freq of L-strategy forecasting')
# inset.plot(t,zHf,HFc,linewidth=1,alpha=.5,label='freq of H-strategy forecasting')
inset.set_xticks([])
inset.set_yticks([])



ax3 = fig2.add_subplot(121)
ax3.set_xlim((0,1))
ax3.set_ylim((0,1))
ax3.set_xticks((0,.2,.4,.6,.8,1))
ax3.set_yticks((0,.2,.4,.6,.8,1))
ax3.set_xticklabels(('0','','','','','1'))
ax3.set_yticklabels(('0','','','','','1'))


ax3.set_xlabel('L-strategy fraction')
ax3.set_ylabel('Environmental state')
ax3.set_title('Long-run strategy-environment dynamics')
ax3.plot(zLm[InvStart:],n[InvStart:],'k-',linewidth=2)


invPoints='yes'
i=0
i2=0
rew=0
ewq=0
for j in range(int(sims)):
    tPost = simOut[j].t
    preEnd = np.argmax(t>tPost[0])
    zLfPost = simOut[j].y[0]
    zHfPost = simOut[j].y[1]
    zLmPost = simOut[j].y[2]
    zHmPost = simOut[j].y[3]
    zLPost = zLmPost + zLfPost
    xLfPost = zLfPost/(zLfPost+zHfPost)
    xLmPost = zLmPost/(zLmPost+zHmPost)
    nPost = simOut[j].y[4]
    if len(t)<len(tPost):
        tPostF = tPost[len(tPost)-len(t):]
        nPostF = nPost[len(nPost)-len(n):]
    else:
        tPostF = tPost[len(tPost)//2:]
        nPostF = nPost[len(nPost)//2:]
    xfPost = zLfPost + zHfPost
    (cycletime,peaks,oneCyclePost)=sb.fourier(tPostF,nPostF)
    pltStrt = np.argmax(tPost>tPost[-1]-Shown*oneCycle)
    tPlotMin = tPost[pltStrt]
    phaseStart = np.argmax(tPost>tPost[-1]-oneCycle)-1
    
    
    if xfPost[-1]>(invFrac):
        ax2.plot(tPost,xfPost,color = sInv,alpha=opac)
        if j%8==0:
            inset.plot(tPost,xfPost,color = sInv,linewidth=1,alpha=1)
        # ax3.plot(xLfPost[phaseStart:],nPost[phaseStart:],color = LFc,linewidth=2,alpha=.3)
        # ax3.plot(xLmPost[phaseStart:],nPost[phaseStart:],color = LMc,linewidth=2,alpha=.3)
    else:
        ax2.plot(tPost,xfPost,color = uInv,alpha=opac)
        if j%8==0:
            inset.plot(tPost,xfPost,color = uInv,linewidth=1,alpha=1)
    if i==0 and xfPost[-1]>(invFrac):
        i=1
        ax3.plot(zLPost[phaseStart:],nPost[phaseStart:],'k-',linewidth=2,alpha=1, label='Population-level')
        ax3.plot(xLfPost[phaseStart:],nPost[phaseStart:],color = fFit,linewidth=2,alpha=.85,label='Forecaster')
        ax3.plot(xLmPost[phaseStart:],nPost[phaseStart:],color = mFit,linewidth=2,alpha=.85,label='Myopic')

        ax2.plot(tPost,xfPost,color = sInv,alpha=opac,label = 'Successful invasion')
    if i2==0 and xfPost[-1]<(invFrac):
        i2=1
        ax2.plot(tPost,xfPost,color = uInv,alpha=opac,label = 'Unsuccessful invasion')

    if invPoints == 'yes' and xfPost[-1]>(invFrac):
        if rew==0:
            rew=1
            ax3.plot(zLPost[0],nPost[0],color = sInv,marker = 'o',markersize=8,linewidth=0,label='Successful invasion point')
        else:
            ax3.plot(zLPost[0],nPost[0],color = sInv, marker = 'o',markersize=8)
    elif invPoints == 'yes' and xfPost[-1]<(invFrac):
        if ewq==0:
            ewq=1
            ax3.plot(zLPost[0],nPost[0],color = uInv,marker = 'o',fillstyle='full',markersize=8,linewidth=0,label='Unsuccessful invasion point')
        else:
            ax3.plot(zLPost[0],nPost[0],color = uInv,marker = 'o',fillstyle='full',markersize=8)
ax2.plot([.2*10000,tEnd+oneCycle,tEnd+oneCycle,.35*10000],[.55,.9*invFrac,.7*invFrac,.05],'k-',linewidth=.75)
ax2.plot([(tInvStart+tEnd+oneCycle)/2,(tInvStart+tEnd+oneCycle)/2],[.7*invFrac,1.05*invFrac],'k-',linewidth=5)
if Linv > 0 and Linv < 1:
    txt = str(Linv)
else:
    txt = 'matched'
    
ax3.legend(loc='upper right')
ax2.legend(loc='upper right')
"Print the parameters on the figure just in case"
paramZ = ('eps1 = '+str(eps1)[0:5]
          +', eps2 = '+str(eps2)[0:5]
          +', r = '+str(r)[0:5]
          +', C = '+str(C)
          +', DT_L1 = '+str(DT_L1)
          +', dt_H0 = '+str(dt_H0)
          +', dt_L0 = '+str(dt_L0)
          +', DT_H1 = '+str(DT_H1)
          +', sims = '+str(sims)                                                                                                                                                                                                                                                                                
          +', popSize = '+str(popSize)
          +', Linv = '+txt
          +', Shown = '+str(Shown)
          +', tEnd = '+str(tEnd)
          +', tFinal = '+str(tFinal)
          +', zLm0 = '+str(zLm0)
          +', n0 = '+str(n0))



if save=='yes' and final != 'yes':
    pars1 = ax3.annotate(paramZ, (0,0), (-30, -30), xycoords='axes fraction', textcoords='offset points', va='top',fontsize=6)
    fig2.savefig('../FIGS/mainTextFig3_'+tag+'.png',dpi=300,bbox_extra_artists=(pars1,), bbox_inches='tight')
if save=='yes' and final == 'yes':
    fig2.savefig('../FIGS/mainTextFig3.png',dpi=300, bbox_inches='tight')

































