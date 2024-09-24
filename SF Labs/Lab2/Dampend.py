#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:45:43 2023

@author: dj-lawton
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

k = 0.2
p = 0.66667
A = 0.0

theta = 0.2
omega = 0.0
t = 0.0
dt = 0.01
timesteps = np.arange(0, 100, 0.01)
THlisttr = []
OMlisttr = []
Nstepstr = []
THlistR = []
OMlistR = []
NstepsR = []

def func(a, b, c):
    return -np.sin(a) - k * b + A*(np.cos(p))*c

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
        THlisttr.append(th)
        OMlisttr.append(om)
        Nstepstr.append(nsteps)

def runge(a, b, c):
    th = a
    om = b
    t = c
    nsteps = 0
    for n in timesteps:
        pa = dt * om
        pb = dt * func(th, om, t)
        
        qa = dt * (om + pb)
        qb = dt * func(th + pa, om + pb, t + dt)
        
        ra =dt * (om + qb/2)
        rb =dt * func(th + qa/2, om + qb/2, t + dt/2)
        
        sa = dt * (om + rb)
        sb = dt * func(th + ra, om + rb, t + dt)
        
        th = th + (pa + 2*qa + 2*ra + sa) / 6
        om = om + (pb + 2*qb + 2*rb + sb ) / 6
        t = n
        nsteps += 1
        THlistR.append(th)
        OMlistR.append(om)
        NstepsR.append(nsteps)




print(func(0.5, 0.7, 6))
trap(3, 0.0, 0)
runge(3, 0.0, 0)

titlefont = {'fontname':'serif', 'size':16}
font = {'fontname':'serif', 'size':12}

plt.figure()
plt.suptitle('Theta and Omega as graphed by the Runge-Kutta Method', fontdict=titlefont)
plt.subplot(211)
plt.ylabel('Theta (radians)', fontdict=font)
plt.axis([0, 100, -np.pi, np.pi])
plt.plot(timesteps, THlistR, '.b', markersize=0.1, label='Rung-Kutta Theta')
plt.legend(loc="upper right")
plt.subplot(212)
plt.ylabel('Omega (radians)', fontdict=font)
plt.xlabel('Time ((10^-2)s)')
plt.axis([0, 100, -np.pi, np.pi])
plt.plot(timesteps, OMlistR, '-y', label='Runge-Kutta Omega')
plt.legend(loc="upper right")
plt.savefig('/home/dj-lawton/Documents/SF Lab Plots/Lab2 Plots/Ex4/THOMGraphs.pdf')
plt.show()

plt.figure()
plt.title('Phase space of a damped system', fontdict=titlefont)
plt.ylabel('Omega(radians)', fontdict=font)
plt.xlabel('Theta(radians)', fontdict=font)
plt.plot(THlistR, OMlistR, '.',markersize=0.5, label='Runge-Kutta Theta')
plt.legend(loc="upper left")
plt.savefig('/home/dj-lawton/Documents/SF Lab Plots/Lab2 Plots/Ex4/THOMGraphDampPhase.pdf')
plt.show()