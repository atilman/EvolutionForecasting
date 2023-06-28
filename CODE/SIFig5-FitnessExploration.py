#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:48:49 2020

@author: atilman
"""
import pylab as plt
import matplotlib as mpl
from scipy.integrate import solve_ivp
import numpy as np
import SimulationBackbone as sb

"GENERAL PARAMS"
eps1 = 15/100;
eps2 = 5/100;
r=5/100;
C = 5/1000;


T1 = 10
R1 = 8
P1 = 6
S1 = 1

R0 = 6
T0 = 2
S0 = 2
P0 = 1



DT_L1 = T1 - R1;
DT_H1 = P1 - S1;
dt_L0 = R0 - T0;
dt_H0 = S0 - P0;



tEndSolo = 1500


zLm00 = 20/100
zLf00 = 20/100
n00 = 20/100
"Nullcline under myopic calculations"


xfrac=np.linspace(0,1,1000)
xNullcline=((dt_H0)*(1-xfrac)+(dt_L0)*xfrac)/((DT_H1+dt_H0)*(1-xfrac)+(DT_L1+dt_L0)*xfrac)
xNullcline[xNullcline>1]=np.nan
xNullcline[xNullcline<0]=np.nan


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



#%%
"One-cycle Average Fitnesses"
t1 = np.arange(tm[0],tm[-1]-2.5*oneCycle,3)

fitAvgM = np.zeros(len(t1))


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
    
#%%
"Computing fitness countours"


x = np.arange(0, 1.005, .005)
n = np.arange(0, 1.005, .005)
X, N = np.meshgrid(x, n)

piL = (1-N)*(R0*X+S0*(1-X))+N*(R1*X+S1*(1-X))
piH = (1-N)*(T0*X+P0*(1-X))+N*(T1*X+P1*(1-X))

FIT = X*piL + (1-X)*piH

#%%
"PLOTS"
"PlotColors"
# import seaborn as sns
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
opac = 1
"PlotWidths"
envWidth = 2
stratWidth = 2
"plotDuration"
tLen = 200
tfEndIndex = np.argmax(tf>tLen)
tmEndIndex = np.argmax(tm>tLen)
tfShow = tf[0:tfEndIndex]
tmShow = tm[0:tmEndIndex]
nmShow = nm[0:tmEndIndex]
nfShow = nf[0:tfEndIndex]
zLmShow = zLmm[0:tmEndIndex]
zHmShow = zHmm[0:tmEndIndex]
zLfShow = zLff[0:tfEndIndex]
zHfShow = zHff[0:tfEndIndex]
fitMyoShow = fitMyo[0:tmEndIndex]
fitForeShow = fitFore[0:tfEndIndex]
"Plots"
fig1 = plt.figure(figsize=(9,9))
ax11 = fig1.add_subplot(2,2,1)
ax11.set_title("Myopic dynamics")
ax11.set_xlabel('Time')
ax11.set_xlim((0,tLen))
ax11.set_ylim((0,1))
ax11.set_yticks((0,.2,.4,.6,.8,1))
ax11.set_yticklabels((0,'','','','',1))
ax11.plot(tmShow,nmShow,color = ENVc,linewidth = envWidth,alpha=1,label='Environmental state')
ax11.plot(tmShow,zLmShow,color = LMc,linewidth = stratWidth,alpha=1,label='L-strategy fraction, mypoic')
ax11.plot(tmShow,zHmShow,color = HMc,linewidth = stratWidth,alpha=.5,label='H-strategy fraction, mypoic')
ax11.legend(loc='upper right')


ax12 = fig1.add_subplot(222)
ax12.set_title('Forecaster dynamics')
ax12.set_xlabel('Time')

ax12.set_ylim((0,1))
ax12.set_yticks((0,.2,.4,.6,.8,1))
ax12.set_yticklabels((0,'','','','',1))
ax12.plot(tfShow,nfShow,color = ENVc,linewidth = envWidth,alpha=1,label='Environmental state')
ax12.plot(tfShow,zLfShow,color = LFc,linewidth = stratWidth,alpha=opac,label='L-strategy fraction, forecaster')
ax12.plot(tfShow,zHfShow,color = HFc,linewidth = stratWidth,alpha=.5,label='H-strategy fraction, forecaster')
ax12.legend(loc='upper right')
ax12.set_xlim(0,tLen)

ax13 = fig1.add_subplot(223)


ax13.set_title('Fitness of forecasting and myopic types')
ax13.set_xlabel('Time')





if len(t1)>5:
    ax13.plot((0,tLen),(fitAvgM[-2],fitAvgM[-2]),'--',color=LMc,label='Myopic average fitness')

    ax13.plot((0,tLen),(fitFore[-1],fitFore[-1]),'--',color=LFc,label='Forecaster average fitness')
else:
    ax13.plot((0,tLen),(fitMyo[-2],fitMyo[-2]),'--',color=LMc,label='Myopic average fitness')
    ax13.plot((0,tLen),(fitFore[-1],fitFore[-1]),'--',color=LFc,label='Forecaster average fitness')
ax13.set_xlim(0,tLen)
ax13.plot(tmShow,fitMyoShow,color=LMc,label='Myopic fitness')
ax13.plot(tfShow,fitForeShow,color=LFc,label='Forecaster fitness')
ax13.legend(loc='upper right')


ax14 = fig1.add_subplot(224)
"add nullclines"
ax14.plot([0,1],[0,1],color="blue") 
ax14.plot(xfrac,xNullcline,color="r") 
ax14.plot([0.004,0.004],[0,1],color="r") 
ax14.plot([.996,.996],[0,1],color="r") 

ax14.set_title("Phase plane")
ax14.set_xlabel('Low-impact fraction')
ax14.set_ylabel('Environmental state')
ax14.set_xlim((0,1))
ax14.set_ylim((0,1))



"fitness colormap"
minColor = np.min((R0,R1,P1,P0))
maxColor = np.max((R0,R1,P1,P0))
levels=np.arange(minColor,maxColor,.1)
cvds = mpl.colormaps['inferno']


cp1=ax14.contourf(X, N, FIT,levels,cmap=cvds,alpha=.65)
cbar1 = fig1.colorbar(cp1)
cbar1.set_label('Fitness', rotation=270)
fig1.tight_layout()

'dynamics in phase plane'
if len(t1)>5:
    tInit = np.argmax(tm>tm[-1]-oneCycle)
    tFin = len(tm)
    ax14.plot(zLmm[tInit:tFin],nm[tInit:tFin],color = HMc,linewidth = stratWidth,alpha=1,label='Myopic limit cycle')
    ax14.plot(zLff[-1],nf[-1],marker='o',color = HFc,linewidth = stratWidth,alpha=1,label='Forecaster equilibrium')
else:
    ax14.plot(zLmm,nm,color = HMc,linewidth = stratWidth,alpha=1,label='Myopic dynamics')
    ax14.plot(zLff,nf,color = HFc,linewidth = stratWidth,alpha=1,label='Forecaster dynamics')
ax14.legend(loc='lower right')





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
          +', tEndSolo = '+str(tEndSolo)
          +', zLm0 = '+str(zLm00)
          +', zLf0 = '+str(zLm00)
          +', n0 = '+str(n00))

save = 'yes'
final = 'yes'

if save=='yes' and final != 'yes':
    pars1 = ax11.annotate(paramZ, (0,0), (-30, -30), xycoords='axes fraction', textcoords='offset points', va='top',fontsize=6)
    fig1.savefig('../FIGS/SIFig5.png',dpi=100,bbox_extra_artists=(pars1,), bbox_inches='tight')
if save=='yes' and final == 'yes':
        fig1.savefig('../FIGS/SIFig5.png',dpi=100, bbox_inches='tight')







