#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:50:16 2023

@author: dj-lawton
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def f(x, n, m):
    if x % (2*np.pi) == 0 or x % (2*np.pi) == np.pi or x == 0:
        g = 0
    elif x % (2*np.pi) <= np.pi:
        g = 1
    else:
        g = -1
    return g


def cosf(k, t, T):
    v = f(t, k, T) * np.cos(t*k*(2*np.pi/T))
    return v


def sinf(k, t, T):
    u = f(t, k, T) * np.sin(t*k*(2*np.pi/T))
    return u


def F(a, b, c, k, T):
    a_b = np.linspace(a, b, 8)
    print(a_b)
    h = (b-a)/len(a_b)
    J1 = 0
    J2 = 0
    for j in range(1, len(a_b)-1):
        if j % 2 == 0:
            J1 += c(a_b[j], k, T)
        else:
            J2 += c(a_b[j], k, T)
        print(c(a, k, T), c(b, k, T))
    F_ = (h/3) * (c(a, k, T) + 2*J1 + 4*J2 + c(b, k, T))
    return F_


def Fourier(T, t):
    Series = (F(0, T, f, 0, T))*(1/T)
# =============================================================================
#     print(Series)
# =============================================================================
    kvals = np.arange(1, 100, 1)
    for k in kvals:
        a = (2/T) * (F(0, T, cosf, k, T))
        print(a)
        b = (2/T) * (F(0, T, sinf, k, T))
        print(b)
        Series += a  * np.cos(k*(2*np.pi)*t/T) + b * np.sin(k*(2*np.pi)*t/T)

    return Series


timesteps = np.arange(0, 2*np.pi, 0.01)
vals = []
for t in timesteps:
    vals.append(Fourier(2*np.pi, t))

valsan = []
for t in timesteps:
    valsan.append(f(t, 0, 2*np.pi))

print(F(0, 2*np.pi, f, 0, 2*np.pi))


plt.figure()
plt.plot(timesteps, vals)
plt.plot(timesteps, valsan)
plt.savefig('/home/dj-lawton/Documents/SF Lab Plots/Lab4 Plots/fourier.pdf')
plt.show()
