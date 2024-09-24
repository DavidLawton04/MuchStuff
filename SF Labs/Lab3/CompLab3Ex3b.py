#!/usr/bin/env python3.
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 03:55:33 2023

@author: dj-lawton
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# spherical grain of dust

Vol = (4/3) * np.pi * ((10**-4)/2)**3


def xmax(rho, D, v, dt, theta):
    Vol = (4/3) * np.pi * (D/2)**3
    m = rho * Vol
    B = 1.6*(10**-4)*D
    timesteps = np.arange(0, 4, dt)

    VyR = v*np.sin(theta)
    VxR = v*np.cos(theta)
    t = 0
    XR = 0
    YR = 0

    for i in range(len(timesteps)-1):
        if YR < 0:
            return XR
            break
        XR = XR + VxR * dt
        YR = YR + VyR * dt
        deltaVyR = -9.81 * dt - (B/m) * VyR * dt
        deltaVxR = -(B/m) * VxR * dt
        VyR = VyR + deltaVyR
        VxR = VxR + deltaVxR
        t = t + dt


timesteps = np.arange(0, 4, 0.0001)
V = 15
theta = np.arange(0.001, np.pi/2, 0.001)


def opt_theta(rho, D, v, dt):
    theta = np.arange(0.01, np.pi/2, 0.001)
    FXvals = []
    for the in theta:
        if the == 0.001:
            X = xmax(rho, D, v, dt, the)
            FXvals.append(X)
        if the != 0.001:
            X = xmax(rho, D, v, dt, the)
            FXvals.append(X)
            if X != max(FXvals):
                return the
                break


rho_vals = np.logspace(0.0, 8.0, num=20, endpoint=True)
opt_t = []
for p in rho_vals:
    opt_t.append(opt_theta(p, 10**-4, 10, 0.00001))

print(opt_t)


plt.figure()
plt.title('Optimum angle vs Mass of a sphere')
plt.xlabel('Mass (kg)')
plt.ylabel('Optimum angle (rads)')
plt.xscale('log')
plt.plot(rho_vals*Vol, opt_t, '.b')
plt.savefig('/home/dj-lawton/Documents/SF Lab Plots/Lab3 Plots/Ex3/opt_t.pdf')
plt.show()
