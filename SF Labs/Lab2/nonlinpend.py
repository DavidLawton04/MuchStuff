#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:50:29 2023

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
timesteps = np.arange(0, 100, 0.01)
THlist = []
OMlist = []
Nsteps = []
THlist2 = []
OMlist2 = []
Nsteps2 = []

def func(a, b, c):
    return -np.sin(a) - k * b + A*(np.cos(p*c))

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

def func2(a, b, c):
    return -a - k * b + A*(np.cos(p*c))

def trap2(a, b, c):
    th = a
    om = b
    t = c
    nsteps = 0
    for n in timesteps:
        pa = dt * om
        pb = dt * func2(th, om, t)
        qa = dt * (om + pb)
        qb = dt * func2(th + pa, om + pb, t + dt)
        
        th = th + (pa + qa) / 2
        om = om + (pb + qb ) / 2
        t = n
        nsteps += 1
        THlist2.append(th)
        OMlist2.append(om)
        Nsteps2.append(nsteps)


titlefont = {'fontname':'serif', 'size':16}
font = {'fontname':'serif', 'size':12}


print(func(0.5, 0.7, 6))
trap(0, 2.0, 0)
trap2(0, 0.5, 0)

plt.figure()
plt.axis([0, 1000, -np.pi, np.pi])
plt.plot(Nsteps, THlist, '-b', label='Theta')
plt.plot(Nsteps, OMlist, '-r', label='Omega')
plt.legend(loc="upper right")
plt.savefig('/home/dj-lawton/Documents/SF Lab Plots/Lab2 Plots/Ex2/THOMvsNsteps5nl.pdf')
plt.show()

plt.figure()
plt.title('Non-Linear vs Linear Pendulum', fontdict = titlefont)
plt.ylabel('Theta (rad)', fontdict = font)
plt.xlabel('Steps', fontdict = font)
plt.axis([0, 1000, -np.pi-1, np.pi+1])
plt.plot(Nsteps, THlist, '-b', label='Non-linear')
plt.plot(Nsteps2, THlist2, '-r', label='Linear')
plt.legend(loc="upper right")
plt.savefig('/home/dj-lawton/Documents/SF Lab Plots/Lab2 Plots/Ex2/lin_vs_non-lin2.pdf')
plt.show()
#Slight phase shifts between linear and non linear, pi produces a straight line
#and 3.14 produces a strange graph due to the very small change in the variables
#sin(theta) ~ 0