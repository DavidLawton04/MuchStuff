#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:12:15 2023

@author: dj-lawton
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# spherical grain of dust
rho = 2*(10**4)
D = 10**-4
Vol = (4/3) * np.pi * (D/2)**3
m = rho * Vol

F = - m * 9.81
print(F)

root1 = 0
root2 = 0

A = 0.25*D**2
B = 1.6*(10**-4)*D
C = F
root1 = (-B + np.sqrt(B ** 2 - 4*A*C)) / (2 * A)
root2 = (-B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)

print(root2)
# take positive root
Kvals = np.arange(0, -1e-04, -1e-06)
f1vals = []
f2vals = []


def f(a, b, c, d):
    p = b * d
    q = c * d**2
    f = p * a + q * (a ** 2)
    return f


def f1(b, c):
    for K in Kvals:
        f1 = b * K
        f2 = c * K**2
        f1vals.append(f1)
        f2vals.append(f2)


f1(B, A)

Vyvals = [0]
timesteps = np.arange(0, 5, 0.0001)


def func(p, q):
    Vy = 0
    t = 0
    dt = 0.0001
    for i in range(len(timesteps)-1):
        deltaVy = -9.81*dt - (p/q)*Vy*dt
        Vy = Vy + deltaVy
        t = t + dt
        Vyvals.append(Vy)


Vyvals2 = []


def func2(p, q):
    t = 0
    dt = 0.0001
    g = 9.81
    for t in timesteps:
        Vy = ((q*g) / p) * (np.exp(-p*t/q)-1)
        t = t + dt
        Vyvals2.append(Vy)


func(B, m)
func2(B, m)
Vyvals2 = np.array(Vyvals2)
Vyvals = np.array(Vyvals)

V_T = -0.642062
print(D*V_T)
# much less than 0.0001, quad term negligible

plt.figure()
plt.xlabel('D * V')
plt.plot(Kvals, f1vals, 'b-', markersize=0.3)
plt.plot(Kvals, f2vals, 'y-', markersize=0.3,)
plt.legend(['Linear', 'Quadratic'], loc='upper left')
plt.savefig('/home/dj-lawton/Documents/SF Lab Plots/Lab3 Plots/Ex2a/Vert1.pdf')
plt.show()
plt.close()
# Clearly quadratic term can be ignored


plt.figure()
plt.xlabel('Time (s)')
plt.ylabel('Vertical velocity (m/s)')
# =============================================================================
# plt.axis([0, 0.26, -1, 0])
# =============================================================================
plt.plot(timesteps, Vyvals2, '-c')
plt.plot(timesteps, Vyvals, '--m')
plt.legend(['Analytic', 'Numerical'])
plt.savefig('/home/dj-lawton/Documents/SF Lab Plots/Lab3 Plots/Ex2b/Vcomp.pdf')
plt.show()
plt.close()

plt.figure()
plt.title('Error of computational vs. analytic solution')
plt.xlabel('Time')
plt.ylabel('Velocity error')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.plot(timesteps, abs(Vyvals2-Vyvals), '-b')
plt.savefig('/home/dj-lawton/Documents/SF Lab Plots/Lab3 Plots/Ex2b/Ecomp.pdf')
plt.show()
