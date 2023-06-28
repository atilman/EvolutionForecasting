#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 10:07:39 2023

@author: atilman
"""
import numpy as np
import pylab as plt
import matplotlib as mpl
from scipy.integrate import solve_ivp
# from scipy.optimize import fsolve
import SimulationBackbone as sb
from multiprocessing import Pool
from matplotlib.colors import LinearSegmentedColormap
import time

def run_simulation(args):
    eps1, eps2, r, C, dt_H0, dt_L0, DT_H1, DT_L1, W0Inv, _, tEnd = args
    return sb.altSysRun([eps1, eps2, r, C, dt_H0, dt_L0, DT_H1, DT_L1, W0Inv, 0, tEnd])

def invasionProb(eps1,eps2,r,C,DT_L1,DT_H1,dt_L0,dt_H0,zLm0,n0,tEnd,tFinal,sims,invFrac):
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
    args = (eps1,eps2,r,C,dt_H0,dt_L0,DT_H1,DT_L1)
    
    zHm0 = 1-zLm0
    W0 = [0, 0, zLm0, zHm0, n0]
    fun = lambda t,y: sb.altSys(y,t,*args)
    sol = solve_ivp(fun,[0,tEnd],W0,rtol=10**-13,atol=10**-14)
    #zLf = sol.y[0]
    #zHf = sol.y[1]
    zLm = sol.y[2]
    zHm = sol.y[3]
    n = sol.y[4]
    t = sol.t
    (cycletimePre,peaksPre,oneCycle)=sb.fourier(t,n)
    "INVASION SIMULATIONS"
    invProb = 0
    tInvStart = t[np.argmax(t>t[-1]-oneCycle)]
    #tInv = np.linspace(tInvStart,tEnd,num=sims)
    tInv = np.arange(tInvStart,tEnd,(tEnd-tInvStart)/sims)
    "INVASION STRUCTURE"
    Invfrac = invFrac
    Lfrac = np.interp(tInv,t,zLm)
    zLmInv = (1-Invfrac)*np.interp(tInv,t,zLm)
    zHmInv = (1-Invfrac)*np.interp(tInv,t,zHm)
    zLfInv = Invfrac*Lfrac
    zHfInv = Invfrac*(1-Lfrac)
    nInv = np.interp(tInv,t,n)
    
    A = np.empty((sims,11),dtype=object)
    # xfMal = np.empty(sims,dtype=object)
    
    
    for i in range(sims):
        W0Inv = [zLfInv[i],zHfInv[i],zLmInv[i],zHmInv[i],nInv[i]]
        A[i] = [eps1,eps2,r,C,dt_H0,dt_L0,DT_H1,DT_L1,W0Inv,tInv[i],tFinal]
    if __name__=='__main__':
        with Pool(7) as pool:        
            simOut = pool.map(sb.altSysRun,A)
    for j in range(sims):
        zLfPost = simOut[j].y[0]
        zHfPost = simOut[j].y[1]
        xfPost = zLfPost + zHfPost
        # xfMal[j] = xfPost[-1]
        if xfPost[-1]>np.min((invFrac,1/200)):
            invProb = invProb + 1/sims
    output = [invProb,DT_L1,DT_H1,dt_L0,dt_H0,sims,eps1,eps2,r,C,zLm0,n0,invFrac,tEnd,tFinal]        
    return(output)


cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
 [0.0589714286, 0.6837571429, 0.7253857143], 
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
 [0.7184095238, 0.7411333333, 0.3904761905], 
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
 [0.9763, 0.9831, 0.0538]]
parula_map = LinearSegmentedColormap.from_list('parula', cm_data)

blu=parula_map(55)
grn=parula_map(135)
yel=parula_map(245)
#%%
"GENERAL PARAMS"
eps1 = 30/100;
eps2 = 10/100; 
r=5/100; 
C = 5/1000; 
tEnd = 800
tFinal = 10000
zLm0 = 80/100
zHm0 = 20/100
n0 = 35/100
sims = 28
invFrac = 1/500



DT_L1 = 2
# DT_H1 = 1 -> 6
# dt_L0 = 1 -> 6
dt_H0 = 1

DT_H1 = np.linspace(2,6,8)
dt_L0 = np.linspace(2,6,10)

invProbData = np.zeros((len(dt_L0),len(DT_H1)))
simStartTime = time.time()

for i in range(len(DT_H1)):
    for j in range(len(dt_L0)):
        runStart = time.time()
        simOutput = invasionProb(eps1, eps2, r, C, DT_L1, DT_H1[i], dt_L0[j], dt_H0, zLm0, n0, tEnd, tFinal, sims, invFrac)
        invProbData[j,i] = simOutput[0]
        print('Run '+str(i)+str(j)+' Done in '+str((time.time()-runStart)/60)+' minutes')
print('Simulation done in '+str((time.time()-simStartTime)/60)+' minutes')           
output = [invProbData,DT_L1,DT_H1,dt_L0,dt_H0,sims,eps1,eps2,r,C,zLm0,n0,invFrac,tEnd,tFinal]
D = np.array(output,dtype=object)
            
np.save('../FIGS/SIFigs-Robustness1.npy', D)
         
#%%            
"PLOTS"

"load data"

Data = np.load('../FIGS/SIFigs-Robustness1.npy',allow_pickle=True)
invProbData = Data[0]
DT_L1 = Data[1]
DT_H1 = Data[2]
dt_L0 = Data[3]
dt_H0 = Data[4]
sims = Data[5]
eps1 = Data[6]
eps2 = Data[7]
r = Data[8]
C = Data[9]
zLm0 = Data[10]   
n0 = Data[11]   
invFrac = Data[12]   
tEnd = Data[13]              
tFinal = Data[14]              
            
            
DT_H1Show, dt_L0Show = np.meshgrid(DT_H1,dt_L0)            
            
fig1 = plt.figure(2)
ax1 = fig1.add_subplot(1, 1, 1)
plt.ylim(1,6)
plt.xlim(1,6)           
            
DS = ax1.contour(DT_H1Show,dt_L0Show,invProbData)
ax1.clabel(DS,DS.levels, fmt='%2.1f',fontsize=12, inline=1)
# ax1.clabel(CS,CS.levels, fmt='$\epsilon_1=\epsilon_{crit}$',fontsize=12, inline=1)


label01='$T_1 - R_1$'
label10='$S_0 - P_0$'
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')

plt.xlabel(label01,fontsize=12)
plt.ylabel(label10,fontsize=12)

x = np.linspace(.01,6,211)
y = np.linspace(.01,6,233)
d01, d10 = np.meshgrid(x,y)
eCrit = (((4*DT_L1*dt_H0+(d01-d10)**2)**(1/2)-d10-d01)*((4*DT_L1*dt_H0+(d01-d10)**2)**(1/2)-2*DT_L1+d01-d10)*((4*DT_L1*dt_H0+(d01-d10)**2)**(1/2)-2*dt_H0-d01+d10))/(8*(DT_L1+d10-d01-dt_H0)**2)
level = [eps1]
level1 = [eps1,eps1+3]

CS = ax1.contour(d01, d10, eCrit,level,colors = 'k')
CS1 = ax1.contourf(d01, d10, eCrit,level1,colors = (yel,blu))
ax1.clabel(CS,CS.levels, fmt='$\epsilon_1=\epsilon_{crit}$',fontsize=12, inline=1)



save = 'yes'
if save =='yes':
    fig1.savefig('../FIGS/SIFigs-Robustness.png',dpi=100, bbox_inches='tight')
       
            