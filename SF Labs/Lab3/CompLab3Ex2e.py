#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:41:00 2023

@author: dj-lawton
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# spherical grain of dust
rho = 2*(10**5)
D = 10**-4
Vol = (4/3) * np.pi * (D/2)**3
m = rho * Vol

F = - m * 9.81
print(F)

A = 0.25*D**2
B = 1.6*(10**-4)*D
C = F

rho_vals = np.logspace(1, 5, 120)


def time(H, p):
    timesteps = np.arange(0, 4, 0.00001)
    VyR = 10
    t = 0
    dt = 0.0001
    YR = 5
    for i in range(len(timesteps)-1):
        if YR < 0:
            break
        YR = YR + VyR * dt
        deltaVyR = -9.81 * dt - (B/(p*Vol)) * VyR * dt
        VyR = VyR + deltaVyR
        t = t + dt
    return t


fall_times = []
for p in rho_vals:
    tfall = time(5, p)
    fall_times.append(tfall)

plt.figure()
plt.xlabel('Mass (kg)')
plt.ylabel('Fall time (s)')
plt.title('Mass vs Time of fall')
plt.plot(rho_vals*Vol, fall_times)
plt.savefig('/home/dj-lawton/Documents/SF Lab Plots/Lab3 Plots/rhovst.pdf')
plt.show()
