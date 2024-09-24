#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:08:38 2023

@author: dj-lawton
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def f(x, n, m):
    g = np.sin(x) + 3*np.sin(3*x) + 5*np.sin(5*x)
    return g


def cosf(k, t, T):
    v = f(t, k, T) * np.cos(t*k*(2*np.pi/T))
    return v


def sinf(k, t, T):
    u = f(t, k, T) * np.sin(t*k*(2*np.pi/T))
    return u


def F(a, b, c, k, T):
    a_b = np.arange(a, b, (b-a)/100)
    n = (len(a_b))
    h = (b-a)/n
    J1 = 0
    J2 = 0
    for j in range(1, len(a_b)):
        if j % 2 == 0:
            J1 += c(k, a_b[j], T)
        else:
            J2 += c(k, a_b[j], T)

    F_ = (h/3) * (f(a, 0, 0) + 2*J1 + 4*J2 + f(b, 0, 0))
    return F_


def Fourier(T, t):
    Series = 0
    kvals = np.arange(0, 10, 1)
    avals = bvals = []
    for k in kvals:
        if k == 0:
            a0 = (F(0, T, f, k, T))*(1/T)
            Series += a0
        if k != 0:
            a = (2/T) * (F(0, T, cosf, k, T))
            avals.append(a)
            b = (2/T) * (F(0, T, sinf, k, T))
            bvals.append(b)
            Series += a * np.cos(k*(2*np.pi)*t/T) + b * np.sin(k*(2*np.pi)*t/T)

    return Series, avals, bvals, a0


def graphing(f, funcstr):
    fig, axs = plt.subplots(3)
    fig.set_figwidth(9)
    fig.set_figheight(15)

    timesteps = np.arange(0, 3.14, 0.01)
    vals = []
    for t in timesteps:
        Series, avals, bvals, a0 = Fourier(2*np.pi, t)
        vals.append(Series)

    valsan = []
    for t in timesteps:
        valsan.append(f(t, 0, 2*np.pi))
    print(avals)

    F(0, np.pi/2, cosf, 1, 2*np.pi)

    axs[0].plot(timesteps, vals, '-m')
    axs[0].set_title('Fourier Series Evaluation')
    axs[1].plot(timesteps, valsan)
    axs[1].set_title('Function')
    axs[2].plot(bvals, '.c', markersize=8)
    axs[2].set_title('Fourier b_n coefficients')
    fig.savefig(f'{funcstr} Series.pdf')


graphing(f, 'sin(x) + 3sin(3x) + 5sin(5x)')
