#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 01:06:15 2023

@author: dj-lawton
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

# spherical grain of dust
rho = 0.5*(10**4)
D = 10**-4
Vol = (4/3) * np.pi * (D/2)**3
m = rho * Vol


A = 0.25*D**2
B = 1.6*(10**-4)*D

timesteps = np.arange(0, 10, 0.001)
t = 0
dt = 0.0001
theta = np.pi/4
v = 5

Vy = v * np.sin(theta)
Vx = v * np.cos(theta)
Vyvals = [Vy]
Vxvals = [Vx]
X = 0
Y = 0
Xvals = [X]
Yvals = [Y]
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

t = 0
VyR = v * np.sin(theta)
VxR = v * np.cos(theta)
VyRvals = [VyR]
VxRvals = [VyR]
XR = 0
YR = 0
XRvals = [XR]
YRvals = [YR]
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

t = 0
VyRq = v * np.sin(theta)
VxRq = v * np.cos(theta)
VyRqvals = [VyRq]
VxRqvals = [VxRq]
XRq = 0
YRq = 0
XRqvals = [XRq]
YRqvals = [YRq]
for i in range(len(timesteps)-1):
    if YRq < 0:
        break
    XRq = XRq + VxRq * dt
    YRq = YRq + VyRq * dt
    Vq = np.sqrt(VxRq ** 2 + VyRq ** 2)
    deltaVyRq = -9.81 * dt - (A/m) * Vq * VyRq * dt
    deltaVxRq = -(A/m) * Vq * VxRq * dt
    VyRq = VyRq + deltaVyRq
    VxRq = VxRq + deltaVxRq
    t = t + dt
    VyRqvals.append(VyRq)
    VxRqvals.append(VxRq)
    XRqvals.append(XRq)
    YRqvals.append(YRq)


plt.figure()
plt.title('Trajectory of spherical particle')
plt.xlabel('X Direction (m)')
plt.ylabel('Y Direction (m)')
plt.plot(Xvals, Yvals, label='Frictionless')
plt.plot(XRvals, YRvals, label='Lin. Approx.')
plt.plot(XRqvals, YRqvals, label='Quad. Approx')
plt.legend(loc='upper left', bbox_to_anchor=(-0.01, 1.01),
           markerscale=2, fancybox=True, framealpha=0.2)
plt.savefig('/home/dj-lawton/Documents/SF Lab Plots/Lab3 Plots/Ex3/ex4tr1.pdf')
plt.show()
