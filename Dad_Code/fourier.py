#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

INT_STEPS = 500
MAX_K = 6


def func(t, x=0, y=0):
    # result = 1
    # result = t
    # result = np.sin(t)
    # result = np.cos(t) + 3 * np.cos(2 * t) - 4 * np.cos(3 * t)
    # result = np.sin(t) + 3 * np.sin(3 * t) + 5 * np.sin(5 * t)
    result = np.sin(t) + 2 * np.cos(3 * t) + 3 * np.sin(5 * t)
    return result


def cosf(t, k, T):
    result = func(t) * np.cos(t * k * (2 * np.pi / T))
    return result


def sinf(t, k, T):
    result = func(t) * np.sin(t * k * (2 * np.pi / T))
    return result


def integrate(f, a, b, k, T, n=INT_STEPS):
    h = (b - a)/n
    a_to_b = np.arange(a, b, h)
    j1 = j2 = 0
    for j in range(1, n):
        if j % 2 == 0:
            j1 += f(a_to_b[j], k, T)
        else:
            j2 += f(a_to_b[j], k, T)

    result = (h/3) * (f(a, 0, T) + (2 * j1) + (4 * j2) + f(b, 0, T))
    return result


def fourier_coeffs(T, maxk=10):
    a = []
    b = []
    for k in np.arange(0, maxk):
        if k == 0:
            a0 = integrate(func, 0, T, k, T) * (1 / T)
        else:
            a.append(integrate(cosf, 0, T, k, T) * (2 / T))
            b.append(integrate(sinf, 0, T, k, T) * (2 / T))
    return a0, a, b


def fourier_series(a0, a, b, t, T, maxk=10):
    series = 0
    for k in np.arange(0, maxk):
        if k == 0:
            series += a0
        else:
            series += (a[k-1] * np.cos(k * (2 * np.pi) * t / T)) + b[k-1] * np.sin(k * (2 * np.pi) * t / T)
    return series


T = 2 * np.pi

print(f"Fourier series, max-k={MAX_K}, int-steps={INT_STEPS}")

func_vals = []
fourier_vals = []

a0, a, b = fourier_coeffs(T, MAX_K)
for k in range(MAX_K):
    if k == 0:
        print(f"a[{k}]={np.round(a0, decimals=4)}")
    else:
        print(f"a[{k}]={np.round(a[k-1], decimals=4)}")
        print(f"b[{k}]={np.round(b[k-1], decimals=4)}")

timesteps = np.arange(0, T, 0.01)
for t in timesteps:
    func_vals.append(func(t))
    fourier_vals.append(fourier_series(a0, a, b, t, T, MAX_K))

plt.figure()
plt.plot(timesteps, func_vals)
plt.plot(timesteps, fourier_vals)
plt.show()
