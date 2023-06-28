#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 10:33:49 2023

@author: atilman
"""
"Here we look at the relative fitness outcomes under myopic cycle vs what would occur at a stable equilibrium. in the case where there are positive incentives to lead cchange in extreme environments"
import numpy as np
import pylab as plt
# import matplotlib as mpl
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import SimulationBackbone as sb
from matplotlib.colors import LinearSegmentedColormap

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

"GENERAL PARAMS"
eps1 = 15/100;
eps2 = 1/100; #not used for the sim
r=5/100; #not used for the sim
C = 5/1000; #not used for the sim
tEnd = 2000
zLm0 = 20/100
n0 = 20/100


T1 = 4
R1 = 2
"P1 from 1 to 7"
S1 = 1

"R0 from 2 to 8"
T0 = 2
S0 = 2
P0 = 1




DT_L1 = T1 - R1
# DT_H1 = P1 - S1
# dt_L0 = R0 - T0
dt_H0 = S0 - P0

d11 = DT_L1
# d01 = DT_H1

d00 = dt_H0
# d10 = dt_L0
#%%
"RUN THIS IF TO REDO SIMULATIONS"
"MYOPIC CYCLES FITNESS"
P1 = np.linspace(S1+1,S1+6,29)
R0 = np.linspace(T0+1.05,T0+6,31)
MyoFit = np.zeros((len(R0),len(P1)))
fit_eq = np.zeros((len(R0),len(P1)))
for i in range(len(P1)):
    for j in range(len(R0)):
        DT_H1 = P1[i] - S1
        dt_L0 = R0[j] - T0
        eCrit1 = (((4*DT_L1*dt_H0+(DT_H1-dt_L0)**2)**(1/2)-dt_L0-DT_H1)*((4*DT_L1*dt_H0+(DT_H1-dt_L0)**2)**(1/2)-2*DT_L1+DT_H1-dt_L0)*((4*DT_L1*dt_H0+(DT_H1-dt_L0)**2)**(1/2)-2*dt_H0-DT_H1+dt_L0))/(8*(DT_L1+dt_L0-DT_H1-dt_H0)**2)
        if DT_H1*dt_L0<DT_L1*dt_H0 or (eps1 > eCrit1 and DT_H1*dt_L0>DT_L1*dt_H0):
            MyoFit[j,i] = np.nan
            fit_eq[j,i] = np.nan
        else:
            payoffDiff = lambda xEQ : ((1-xEQ)*(R0[j]*xEQ+S0*(1-xEQ))+xEQ*(R1*xEQ+S1*(1-xEQ)))-((1-xEQ)*(T0*xEQ+P0*(1-xEQ))+xEQ*(T1*xEQ+P1[i]*(1-xEQ))) 
            xEQ_initial_guess = 0.5
            x_eq = fsolve(payoffDiff, xEQ_initial_guess)
            fit_eq[j,i] = ((1-x_eq)*(R0[j]*x_eq+S0*(1-x_eq))+x_eq*(R1*x_eq+S1*(1-x_eq)))*x_eq + (1-x_eq)*((1-x_eq)*(T0*x_eq+P0*(1-x_eq))+x_eq*(T1*x_eq+P1[i]*(1-x_eq)))

            "SIM: MYOPIC ONLY"
            
            args = (eps1,eps2,r,C,dt_H0,dt_L0,DT_H1,DT_L1)
            zHm0 = 1-zLm0
            W0 = [0, 0, zLm0, zHm0, n0]
            fun = lambda t,y: sb.altSys(y,t,*args)
            sol = solve_ivp(fun,[0,tEnd],W0,rtol=10**-10,atol=10**-10)
            zLm = sol.y[2]
            zHm = sol.y[3]
            nm = sol.y[4]
            tm = sol.t
            "Instantaneous Fitnesses"
            "only Myopic"
            piL = (1-nm)*(R0[j]*zLm+S0*zHm)+nm*(R1*zLm+S1*zHm)
            piH = (1-nm)*(T0*zLm+P0*zHm)+nm*(T1*zLm+P1[i]*zHm)
            fitMyo = zLm*piL + zHm*piH
            "One-cycle Average Fitnesses"
            (cycletimePre,peaksPre,oneCycle)=sb.fourier(tm,nm)
            t1 = np.linspace(tEnd-6*oneCycle,tEnd-3*oneCycle,5)
            fitAvgM = np.zeros(len(t1))
            if oneCycle<tEnd/7:
                for u in range (len(t1)):
                    Start = np.argmax(tm>t1[u])
                    for k in range(Start+1,len(tm)-1,1):
                        if nm[k] > nm[k+1] and nm[k] > nm[k-1]:
                            Start1 = k
                            Mal1 = len(tm)-1
                            for l in range(Start1+10,len(tm)-1,1):
                                if nm[l] > nm[l+1] and nm[l] > nm[l-1]:
                                    Mal1 = l
                                    break
                            break
                    fitAvgM[u]= sb.avg(tm[Start1:Mal1],fitMyo[Start1:Mal1])
                    t1[u] = tm[Start1]
            if oneCycle>tEnd/7:
                fitAvgM[-1]=np.nan
            MyoFit[j,i] = fitAvgM[-1]
    
output = [MyoFit,fit_eq,eps1,T1,R1,P1,S1,R0,T0,S0,P0]
D = np.array(output,dtype=object)

np.save('../FIGS/SIFig6a-FitnessTest.npy', D)

#%%


"PLOTS"

"load data"

Data = np.load('../FIGS/SIFig6a-FitnessTest.npy',allow_pickle=True)
MyoFit = Data[0]
fit_eq = Data[1]
eps1 = Data[2]
T1 = Data[3]
R1 = Data[4]
P1 = Data[5]
S1 = Data[6]
R0 = Data[7]
T0 = Data[8]
S0 = Data[9]
P0 = Data[10]

d11 = T1 - R1
d00 = S0 - P0

x = np.linspace(.01,6,211)
y = np.linspace(.01,6,233)



fig1 = plt.figure(2)
ax1 = fig1.add_subplot(1, 1, 1)
plt.ylim(1,6)
plt.xlim(1,6)


d01, d10 = np.meshgrid(x,y)
DT_H1Show, dt_L0Show = np.meshgrid(P1-S1,R0-T0)
eCrit = (((4*d11*d00+(d01-d10)**2)**(1/2)-d10-d01)*((4*d11*d00+(d01-d10)**2)**(1/2)-2*d11+d01-d10)*((4*d11*d00+(d01-d10)**2)**(1/2)-2*d00-d01+d10))/(8*(d11+d10-d01-d00)**2)
levels=[eps1,eps1+3]
level = [eps1]
CS = ax1.contour(d01, d10, eCrit,level,colors = 'k')
CS1 = ax1.contourf(d01, d10, eCrit,levels,colors = (yel,blu))
DS = ax1.contour(DT_H1Show,dt_L0Show,fit_eq-MyoFit)
ax1.clabel(DS,DS.levels, fmt='%2.1f',fontsize=12, inline=1)
ax1.clabel(CS,CS.levels, fmt='$\epsilon_1=\epsilon_{crit}$',fontsize=12, inline=1)
# ax1.fill_between(d01,5,CS.levels[0],facecolor='b')


label01='$T_1 - R_1$'
label10='$S_0 - P_0$'

ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')

plt.xlabel(label01,fontsize=12)
plt.ylabel(label10,fontsize=12)

save = 'yes'
if save =='yes':
    fig1.savefig('../FIGS/SIFig6a.png',dpi=100, bbox_inches='tight')










