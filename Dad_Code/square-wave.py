#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

INT_STEPS = 500
MAX_K = 100

def func(t, x=0, y=0):
    if t % (2*np.pi) == 0 or t % (2*np.pi) == np.pi or t == 0:
        g = 0
    elif t % (2*np.pi) <= np.pi:
        g = 1
    else:
        g = -1
    return g
# =============================================================================
#     if 0 <= t <= np.pi:
#         result = 1
#     else:
#         result = -1
#     return result
# =============================================================================


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

print(f"Square wave, max-k={MAX_K}, int-steps={INT_STEPS}")

a0, a, b = fourier_coeffs(T, MAX_K)
for k in range(MAX_K):
    if k == 0:
        print(f"a[{0}]={np.round(a0, decimals=4)} expected 0")
    else:
        print(f"a[{k}]={np.round(a[k-1], decimals=4)} expected 0")
        if k % 2 == 0:
            print(f"b[{k}]={np.round(b[k-1], decimals=4)} expected 0")
        else:
            print(f"b[{k}]={np.round(b[k-1], decimals=4)} expected {4 / np.pi / k}")

func_vals = []
func_series10 = []
func_series100 = []
fourier_vals = {}

timesteps = np.arange(0, 2*T, 0.01)
for t in timesteps:
    func_vals.append(func(t))
    func_series10.append(fourier_series(a0, a, b, t, T, maxk=10))
    func_series100.append(fourier_series(a0, a, b, t, T, maxk=100))

for i in (1, 2, 3, 5, 10, 20, 30):
    fourier_vals[i] = []
    for t in timesteps:
        fourier_vals[i].append(fourier_series(a0, a, b, t, T, i))

np.array(func_vals)
np.array(func_series10)

plt.figure('error')
plt.plot(timesteps, func_vals-func_series10)
plt.show()

plt.figure('Square Wave Series', figsize=(10, 4))
plt.plot(timesteps, func_vals)
for i in (1, 2, 3, 5, 10, 20, 30):
    plt.plot(timesteps, fourier_vals[i])
plt.savefig("square-wave.pdf")
plt.show()
plt.close()
