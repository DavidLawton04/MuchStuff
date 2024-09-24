#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:33:22 2023

@author: dj-lawton
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

k = 0.5
p = 0.66667
A = 1.5
t = 0.0
dt = 0.01
timesteps = np.arange(0, 10000, 0.01)
iteration_number = 0
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
    for n in timesteps:
        pa = dt * om
        pb = dt * func(th, om, t)
        qa = dt * (om + pb)
        qb = dt * func(th + pa, om + pb, t + dt)
        
        th = th + (pa + qa) / 2
        om = om + (pb + qb ) / 2
        t = n
        if (np.abs(th) > np.pi):
            th -= 2 * np.pi * np.abs(th) / th
        nsteps += 1
        THlisttr.append(th)
        OMlisttr.append(om)
        Nstepstr.append(nsteps)

def runge(a, b, c):
    th = a
    om = b
    t = c
    iteration_number = 0
    transient=5000
    for n in range(220000):
        pa = dt * om
        pb = dt * func(th, om, t)
        
        qa = dt * (om + pb/2)
        qb = dt * func(th + pa/2, om + pb/2, t + dt/2)
        
        ra =dt * (om + qb/2)
        rb =dt * func(th + qa/2, om + qb/2, t + dt/2)
        
        sa = dt * (om + rb)
        sb = dt * func(th + ra, om + rb, t + dt)
        
        th = th + (pa + 2*qa + 2*ra + sa) / 6
        om = om + (pb + 2*qb + 2*rb + sb ) / 6
        if (np.abs(th) > np.pi):
            th -= 2 * np.pi * np.abs(th) / th
        t = t+dt
        iteration_number += 1
        if iteration_number > transient:
            THlistR.append(th)
            OMlistR.append(om)
            NstepsR.append(iteration_number)




print(func(0.5, 0.7, 6))
trap(3.14, 0.0, 0)
runge(3.14, 0.0, 0)

titlefont = {'fontname':'serif', 'size':16}
font = {'fontname':'serif', 'size':12}

print(THlistR)
print(OMlistR)
plt.figure()
plt.title('Phase space of a driven, damped, non-linear pendulum.', fontdict=titlefont)
plt.ylabel('Omega(radians)', fontdict=font)
plt.xlabel('Theta(radians)', fontdict=font)
plt.plot(THlistR, OMlistR, '.r',markersize=0.5, label='Runge-Kutta Theta')
plt.legend(loc="upper left")
# =============================================================================
# plt.savefig('/home/dj-lawton/Documents/SF Lab Plots/Lab2 Plots/Ex5/THOMGraph5b.pdf')
# =============================================================================
plt.show()
