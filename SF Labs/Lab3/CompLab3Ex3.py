#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 09:43:18 2023

@author: dj-lawton
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#spherical grain of dust
rho = 2*(10**5)
D = 10**-4
Vol = (4/3) * np.pi * (D/2)**3
m = rho * Vol

F = - m * 9.81
print(F)

A = 0.25*D**2
B = 1.6*(10**-4)*D
C = F


timesteps = np.arange(0, 4, 0.0001)

VyRvals = [10]
VxRvals = [4]
VyR = 10
VxR = 4
t = 0
dt = 0.0001
XR = 0
YR = 0
XRvals = [0]
YRvals = [0]
for i in range(len(timesteps)-1):
    if YR < 0:
        break
    XR = XR + VxR * dt
    YR = YR + VyR * dt
    deltaVyR = -9.81 * dt - (B/m) * VyR * dt
    deltaVxR = -(B/m) * VxR * dt
    VyR = VyR + deltaVyR
    VxR = VxR + deltaVxR
    t = t + dt
    VyRvals.append(VyR)
    VxRvals.append(VxR)
    XRvals.append(XR)
    YRvals.append(YR)

Vyvals = [10]
Vxvals = [4]
Vy = 10
Vx = 4
t = 0
dt = 0.0001
X = 0
Y = 0
Xvals = [0]
Yvals = [0]
for i in range(len(timesteps)-1):
    if Y < 0:
        break
    X = X + Vx * dt
    Y = Y + Vy * dt
    deltaVy = -9.81 * dt
    Vy = Vy + deltaVy
    t = t + dt
    Vyvals.append(Vy)
    Xvals.append(X)
    Yvals.append(Y)

plt.figure()
plt.title('Trajectory of spherical particle')
plt.xlabel('X Direction (m)')
plt.ylabel('Y Direction (m)')
plt.plot(Xvals, Yvals)
plt.plot(XRvals, YRvals)
plt.savefig('/home/dj-lawton/Documents/SF Lab Plots/Lab3 Plots/Ex3/traj1.pdf')
plt.show()
