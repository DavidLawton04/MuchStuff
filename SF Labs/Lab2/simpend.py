#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:52:10 2023

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
timesteps = np.arange(0, 100, 0.001)
THlist = []
OMlist = []
Nsteps = []

def func(a, b, c):
    return -a - k * b + A*(np.cos(p*c))

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
        nsteps += 1
        THlist.append(th)
        OMlist.append(om)
        Nsteps.append(nsteps)




print(func(0.5, 0.7, 6))
trap(0, 2.0, 0)

titlefont = {'fontname':'serif', 'size':16}
font = {'fontname':'serif', 'size':12}

plt.figure()
plt.axis([0, 1000, -np.pi, np.pi])
plt.plot(Nsteps, THlist, '-b', markersize=0.5, label='Theta')
plt.plot(Nsteps, OMlist, '-r', markersize=0.5, label='Omega')
plt.legend(loc="upper right")
plt.savefig('/home/dj-lawton/Documents/SF Lab Plots/Lab2 Plots/Ex1/THOM_vs_steps5.pdf')
plt.show()
