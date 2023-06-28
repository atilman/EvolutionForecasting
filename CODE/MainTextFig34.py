#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:48:49 2020

@author: atilman
"""
import pylab as plt

from scipy.integrate import solve_ivp
import numpy as np
import SimulationBackbone as sb


"GENERAL PARAMS"
eps1 = 3/10;
eps2 = 1/10;
r=15/100;
C = 5/1000;
R0 = 5
R1 = 0
S0 = 2
S1 = 0
T0 = 0
T1 = 2
P0 = 0
P1 = 4

DT_L1 = T1 - R1;
DT_H1 = P1 - S1;
dt_L0 = R0 - T0;
dt_H0 = S0 - P0;


Shown = 6
popSize=70
invFrac=1/popSize

tEndSolo = 700
tEndInv = 5050

zLm00 = 3/10
zLf00 = 3/10
n00 = 9/10

"SIM: NO forecasters"

args = (eps1,eps2,r,C,dt_H0,dt_L0,DT_H1,DT_L1)
zHm00 = 1-zLm00
W0 = [0, 0, zLm00, zHm00, n00]
fun = lambda t,y: sb.altSys(y,t,*args)
sol = solve_ivp(fun,[0,tEndSolo],W0,rtol=10**-13,atol=10**-14)
zLmm = sol.y[2]
zHmm = sol.y[3]
nm = sol.y[4]
tm = sol.t
(cycletimePre,peaksPre,oneCycle)=sb.fourier(tm,nm)

"SIM: ONLY forecasters"
args = (eps1,eps2,r,C,dt_H0,dt_L0,DT_H1,DT_L1)
zHf00 = 1-zLf00
W0 = [zLf00, zHf00, 0, 0, n00]
fun = lambda t,y: sb.altSys(y,t,*args)
sol = solve_ivp(fun,[0,tEndSolo],W0,rtol=10**-13,atol=10**-14)
zLff = sol.y[0]
zHff = sol.y[1]
nf = sol.y[4]
tf = sol.t

"SIM: Invasion"
zL0 = zLmm[-1]
zLm0 = zL0*(1-invFrac)
zHm0 = (1-zL0)*(1-invFrac)
zLf0 = zL0*invFrac
zHf0 = (1-zL0)*invFrac
n0 = nm[-1]

args = (eps1,eps2,r,C,dt_H0,dt_L0,DT_H1,DT_L1)

W0 = [zLf0, zHf0, zLm0, zHm0, n0]
fun = lambda t,y: sb.altSys(y,t,*args)
sol = solve_ivp(fun,[0,tEndInv],W0,rtol=10**-13,atol=10**-14)
zLf = sol.y[0]
zHf = sol.y[1]
zLm = sol.y[2]
zHm = sol.y[3]
n = sol.y[4]
t = sol.t

#%%
"Instantaneous Fitnesses"
"only Myopic"
piL = (1-nm)*(R0*zLmm+S0*zHmm)+nm*(R1*zLmm+S1*zHmm)
piH = (1-nm)*(T0*zLmm+P0*zHmm)+nm*(T1*zLmm+P1*zHmm)
fitMyo = zLmm*piL + zHmm*piH

"Only Forecaster"
piL = (1-nf)*(R0*zLff+S0*zHff)+nf*(R1*zLff+S1*zHff)
piH = (1-nf)*(T0*zLff+P0*zHff)+nf*(T1*zLff+P1*zHff)
fitFore = zLff*piL + zHff*piH

"Both under invasion situation"
zL = zLf + zLm
zH = zHf + zHm
piL = (1-n)*(R0*zL+S0*zH)+n*(R1*zL+S1*zH)
piH = (1-n)*(T0*zL+P0*zH)+n*(T1*zL+P1*zH)
fitInvM = (zLm/(zLm+zHm))*piL + (zHm/(zLm+zHm))*piH
fitInvF = (zLf/(zLf+zHf))*piL + (zHf/(zLf+zHf))*piH

#%%
"One-cycle Average Fitnesses"
t1 = np.arange(tm[0],tm[-1]-2.5*oneCycle,3)
t2 = np.arange(t[0]+2*oneCycle,t[-1]-3*oneCycle,10)
fitAvgM = np.zeros(len(t1))
fitAvgInvM = np.zeros(len(t2))
fitAvgInvF = np.zeros(len(t2))

for i in range (len(t1)):
    Start = np.argmax(tm>t1[i])
    for k in range(Start+1,len(tm)-1,1):
        if nm[k] > nm[k+1] and nm[k] > nm[k-1]:
            Start1 = k
            Mal1 = len(tm)-1
            for l in range(Start1+10,len(tm)-1,1):
                if nm[l] > nm[l+1] and nm[l] > nm[l-1]:
                    Mal1 = l
                    break
            break
    fitAvgM[i]= sb.avg(tm[Start1:Mal1],fitMyo[Start1:Mal1])
    t1[i] = tm[Start1]
    
for i in range (len(t2)):
    Start = np.argmax(t>t2[i])
    for k in range(Start+1,len(t)-1,1):
        if n[k] > n[k+1] and n[k] > n[k-1]:
            Start1 = k
            Mal1 = len(t)-1
            for l in range(Start1+10,len(t)-1,1):
                if n[l] > n[l+1] and n[l] > n[l-1]:
                    Mal1 = l
                    break
            break
    fitAvgInvM[i]= sb.avg(t[Start1:Mal1],fitInvM[Start1:Mal1])
    fitAvgInvF[i]= sb.avg(t[Start1:Mal1],fitInvF[Start1:Mal1]) - C
    t2[i] = t[Start1]
#%%
"PLOTS"
"PlotColors"
forecasterColor = "darkorchid"             
myopicColor = "darkcyan"
ENVc = 'gold'

LMc = (0.375, 0.625, 0.625)
HMc = (0.0, 0.5450980392156861, 0.5450980392156862)
LFc = (0.5568983957219252, 0.32372549019607844, 0.6723529411764706)
HFc = (0.6662083015024193, 0.0, 0.996078431372549)
mFit = (0.0953921568627451, 0.449705882352941, 0.4497058823529411)
fFit = (0.593895594601477, 0.21415686274509815, 0.7819215686274509)
sInv = (0.8235294117647058, 0.0, 0.0)
uInv = (0.4941176470588235, 0.32941176470588235, 0.32941176470588235)

opac = .5
"PlotWidths"
envWidth = 2
stratWidth = 2
"plotDuration"
tLen = 50

"Plots"
fig1 = plt.figure(figsize=(15,5))
ax11 = fig1.add_subplot(1,3,1)
ax11.set_title("Myopic dynamics")
ax11.set_xlabel('Time')
ax11.set_xlim((0,tLen))
ax11.set_ylim((0,1))
ax11.set_yticks((0,.2,.4,.6,.8,1))
ax11.set_yticklabels((0,'','','','',1))
ax11.plot(tm,nm,color = ENVc,linewidth = envWidth,alpha=1,label='Environmental state')
ax11.plot(tm,zLmm,color = LMc,linewidth = stratWidth,alpha=opac,label='L-strategy fraction, mypoic')
ax11.plot(tm,zHmm,color = HMc,linewidth = stratWidth,alpha=opac,label='H-strategy fraction, mypoic')
ax11.legend(loc='upper right')

ax12 = fig1.add_subplot(132)
ax12.set_title('Forecaster dynamics')
ax12.set_xlabel('Time')
ax12.set_xlim(0,tLen)
ax12.set_ylim((0,1))
ax12.set_yticks((0,.2,.4,.6,.8,1))
ax12.set_yticklabels((0,'','','','',1))
ax12.plot(tf,nf,color = ENVc,linewidth = envWidth,alpha=1,label='Environmental state')
ax12.plot(tf,zLff,color = LFc,linewidth = stratWidth,alpha=opac,label='L-strategy fraction, forecaster')
ax12.plot(tf,zHff,color = HFc,linewidth = stratWidth,alpha=opac,label='H-strategy fraction, forecaster')
ax12.legend(loc='upper right')

ax13 = fig1.add_subplot(133)
ax13.set_title('Coexistence of myopic and forecasting types')
ax13.set_xlabel('Time')
ax13.set_xlim(tEndInv-tLen,tEndInv)
ax13.set_ylim((0,1))
ax13.set_yticks((0,.2,.4,.6,.8,1))
ax13.set_yticklabels((0,'','','','',1))
ax13.plot(t,n,color = ENVc,linewidth = envWidth,alpha=1,label='Environmental state')
ax13.plot(t,zLm,color = LMc,linewidth = stratWidth,alpha=opac,label='L-strategy fraction, mypoic')
ax13.plot(t,zHm,color = HMc,linewidth = stratWidth,alpha=opac,label='H-strategy fraction, mypoic')
ax13.plot(t,zLf,color = LFc,linewidth = stratWidth,alpha=opac,label='L-strategy fraction, forecaster')
ax13.plot(t,zHf,color = HFc,linewidth = stratWidth,alpha=opac,label='H-strategy fraction, forecaster')
ax13.legend(loc='upper right')



fig2 = plt.figure(figsize=(10,10))
fig2.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.33)
plotEndm = np.argmax(tm>Shown*oneCycle)
insetT = .43
insetLoc = .3
tEndInv1 = 3000
ax21 = fig2.add_subplot(211)
ax21.set_title("Successful invasion dynamics")
ax21.set_xlabel('Time')
ax21.set_xlim((0,tEndInv1))
ax21.set_ylim((0,1))
ax21.set_yticks((0,.2,.4,.6,.8,1))
ax21.set_yticklabels((0,'','','','',1))
ax21.plot(t,n,color = ENVc,linewidth = envWidth,alpha=1,label='Environmental state')
ax21.plot(t,zLm,color = LMc,linewidth = stratWidth,alpha=opac,label='L-strategy fraction, mypoic')
ax21.plot(t,zHm,color = HMc,linewidth = stratWidth,alpha=opac,label='H-strategy fraction, mypoic')
ax21.plot(t,zLf,color = LFc,linewidth = stratWidth,alpha=opac,label='L-strategy fraction, forecaster')
ax21.plot(t,zHf,color = HFc,linewidth = stratWidth,alpha=opac,label='H-strategy fraction, forecaster')
ax21.plot([.24*tEndInv1,insetT*tEndInv1+.5*Shown*oneCycle,insetT*tEndInv1+.5*Shown*oneCycle,.27*tEndInv1],[.7,.3,.3,.95],'k-',linewidth=.75)
ax21.plot([insetT*tEndInv1+.5*Shown*oneCycle,insetT*tEndInv1+.5*Shown*oneCycle],[.02,.65],'k-',linewidth=2)
ax21.legend(loc='upper right')

inset = plt.axes([insetLoc, 0.77, .1, .1])
inset.set_xlim((insetT*tEndInv1,insetT*tEndInv1+.5*Shown*oneCycle))
inset.set_ylim((0,.7))
inset.plot(t,n,color = ENVc,linewidth = envWidth,alpha=1,label='Environmental state')
inset.plot(t,zLm,color = LMc,linewidth=stratWidth,alpha=opac,label='L-strategy fraction, mypoic')
inset.plot(t,zHm,color = HMc,linewidth=stratWidth,alpha=opac,label='H-strategy fraction, mypoic')
inset.plot(t,zLf,color = LFc,linewidth=stratWidth,alpha=opac,label='L-strategy fraction, forecaster')
inset.plot(t,zHf,color = HFc,linewidth=stratWidth,alpha=opac,label='H-strategy fraction, forecaster')
inset.set_xticks([])
inset.set_yticks([])




ax22 = fig2.add_subplot(212)
ax22.set_title("Invasion fitness")
ax22.set_xlabel("Time")
ax22.set_ylabel('Fitness')
ax22.set_yticks((1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75))
ax22.set_yticklabels(('',1.4,'',1.5,'',1.6,'',1.7,''))
ax22.plot(t2,fitAvgInvM,color = mFit,linewidth=2,alpha=opac,label='Myopic fitness')
ax22.plot(t2,fitAvgInvF,color = fFit,linewidth=2,alpha=opac,label='Forecaster fitness')


ax22.plot((0,t[-1]),(fitAvgM[-1],fitAvgM[-1]),linestyle = 'dashed', color = HMc,label='Myopic-only long-run fitness')
ax22.plot((0,t[-1]),(fitFore[-1] - C,fitFore[-1] - C),linestyle = 'dashed',color = HFc, label='Forecaster-only long-run fitness')
ax22.legend(loc='upper right')

"set axes bounds"
invMax = np.max(fitAvgInvF)
yMin = np.min(fitAvgM) - .1*(np.max((fitFore[-1],invMax)) - np.min(fitAvgM))
yMax = np.max((fitFore[-1],invMax)) + .1*(np.max((fitFore[-1],invMax)) - np.min(fitAvgM))
ax22.set_ylim((yMin,yMax))
ax22.set_ylim((1.35,1.75))
ax22.set_xlim((0,tEndInv1))

"Print the parameters on the figure just in case"
paramZ = ('eps1 = '+str(eps1)[0:5]
          +', eps2 = '+str(eps2)[0:5]
          +', r = '+str(r)[0:5]
          +', C = '+str(C)
          +', R0 = '+str(R0)
          +', R1 = '+str(R1)
          +', S0 = '+str(S0)
          +', S1 = '+str(S1)
          +', T0 = '+str(T0)
          +', T1 = '+str(T1)
          +', P0 = '+str(P0)
          +', P1 = '+str(P1)
          +', Shown = '+str(Shown)
          +', popSize = '+str(popSize)
          +', tEndSolo = '+str(tEndSolo)
          +', tEndInv = '+str(tEndInv)
          +', zLm0 = '+str(zLm00)
          +', zLf0 = '+str(zLm00)
          +', n0 = '+str(n00))

save = 'yes'
final = 'yes'

if save=='yes' and final != 'yes':
    pars1 = ax11.annotate(paramZ, (0,0), (-30, -30), xycoords='axes fraction', textcoords='offset points', va='top',fontsize=6)
    pars2 = ax22.annotate(paramZ, (0,0), (-30, -30), xycoords='axes fraction', textcoords='offset points', va='top',fontsize=6)
    fig1.savefig('../FIGS/Fig3.png',dpi=300,bbox_extra_artists=(pars1,), bbox_inches='tight')
    fig2.savefig('../FIGS/Fig4.png',dpi=300,bbox_extra_artists=(pars1,), bbox_inches='tight')
if save=='yes' and final == 'yes':
        fig1.savefig('../FIGS/Fig3.png',dpi=300, bbox_inches='tight')
        fig2.savefig('../FIGS/Fig4.png',dpi=300, bbox_inches='tight')





