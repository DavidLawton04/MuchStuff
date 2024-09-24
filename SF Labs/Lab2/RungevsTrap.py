#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:18:52 2023

@author: dj-lawton
"""

import numpy as np
import matplotlib.pyplot as plt

k = 0.0
p = 0.66667
A = 0.0

theta = 0.2
omega = 0.0
t = 0.0
dt = 0.01
timesteps = np.arange(0, 10000, 0.01)
THlisttr = []
OMlisttr = []
Nstepstr = []
THlistR = []
OMlistR = []
NstepsR = []

def func(a, b, c):
    return -np.sin(a) - k*b + A*(np.cos(p*c))

def trap(a, b, c):
    th = a
    om = b
    t = c
    nsteps = 0
    for n in range(len(timesteps)):
        pa = dt * om
        pb = dt * func(th, om, t)
        qa = dt * (om + pb)
        qb = dt * func(th + pa, om + pb, t + dt)
        
        th = th + (pa + qa) / 2
        om = om + (pb + qb) / 2
        t = t+dt
        nsteps += 1
        THlisttr.append(th)
        OMlisttr.append(om)
        Nstepstr.append(nsteps)

def runge(a, b, c):
    th = a
    om = b
    t = c
    nsteps = 0
    for n in range(len(timesteps)):
        pa = dt * om
        pb = dt * func(th, om, t)
        
        qa = dt * (om + pb/2)
        qb = dt * func(th + pa/2, om + pb/2, t + dt/2)
        
        ra =dt * (om + qb/2)
        rb =dt * func(th + qa/2, om + qb/2, t + dt/2)
        
        sa = dt * (om + rb)
        sb = dt * func(th + ra, om + rb, t + dt)
        
        th = th + (pa + 2*qa + 2*ra + sa) / 6
        om = om + (pb + 2*qb + 2*rb + sb) / 6
        t = n
        nsteps += 1
        THlistR.append(th)
        OMlistR.append(om)
        NstepsR.append(nsteps)




print(func(0.5, 0.7, 6))
trap(3.14159, 0.0, 0)
runge(np.pi, 0.0, 0)

titlefont = {'fontname':'serif', 'size':16}
font = {'fontname':'serif', 'size':12}

plt.figure()
plt.axis([0, 500, -np.pi-1, np.pi+1])
plt.ylabel('Theta', fontdict=font)
plt.xlabel('Steps', fontdict=font)
plt.plot(timesteps, THlisttr, '.b', label='Tr Theta')
plt.plot(timesteps, THlistR, '.r', label='R-K Theta')
plt.legend(loc="upper right")
plt.savefig('/home/dj-lawton/Documents/SF Lab Plots/Lab2 Plots/Ex3/RungvsTrapnl.pdf')
plt.show()