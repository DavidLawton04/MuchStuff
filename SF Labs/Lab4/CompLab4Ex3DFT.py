#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

INT_STEPS = 500
MAX_K = 500

def func(t, x=0, y=0):
    if t % (2*np.pi) <= np.pi:
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


def DFT_coeffs(funclist, N):
    Fnreal_vals = []
    Fnimag_vals = []
    kvals = np.arange(0, maxk)
    for k in kvals:
        nreal = 0
        nimag = 0
        for m in range(0, N):
            Fnreal += func_vals[m] * np.cos(2 * np.pi * m * k / N)
            Fnimag += -func_vals[m] * np.sin(2 * np.pi * m * k / N)
        Fnreal_vals.append(Fnreal)
        Fnimag_vals.append(Fnimag)
    return Fnreal_vals, Fnimag_vals, kvals


def fourier_series(a0, a, b, t, T, maxk=10):
    series = 0
    for k in np.arange(0, maxk):
        if k == 0:
            series += a0
        else:
            series += (a[k-1] * np.cos(k * (2 * np.pi) * t / T)) + b[k-1] * np.sin(k * (2 * np.pi) * t / T)
    return series


def funcvals(N, h, f, funcstr):
    Tau = N * h

    print(f"\nDFT, N={N} h={h}")
    print(f"Sample frequency: {1/h} Hz")

    omega1 = 2 * np.pi / Tau
    omegaN = 1 / (2 * h) - 1 / (N * h)

    print(f"Fundamental frequency: {omega1}")
    print(f"Nyquist frequency: {omegaN}")    

    # Ideal sampling interval
    freq = 6 * np.pi / (2 * np.pi)
    period = 1 / freq
    print(f"Actual frequency: {freq} Hz")
    print(f"Actual period: {period} s")

    actual_func_vals = []
    timesteps = np.arange(0, N * h, 0.001)
    for i in timesteps:
        actual_func_vals.append(func(i))

    func_vals = []
    time_vals = []
    samples = range(0, N)
    for m in samples:
        time_vals.append(m*h)
        func_vals.append(f(m*h))
    Fnreal_vals, Fnimag_vals, kvals = DFT_coeffs(func_vals, N)

    fig, axs = plt.subplot(2, 2)
    fig.setwidth(15)
    fig.setheight(10)
    fig.suptitle(f"DFT, func={funcstr}, N={N}, h={h}")

    axs[0].plot(timesteps, actual_func_vals, '-b', label='acfunc')
    axs[0].xlabel('Time')
    axs[0].title('Actual Function')

    axs[1].plot(samples, func_vals, drawstyle='steps-pre', label='Function')
    axs[1].plot(samples, func_vals, '.', label='Function samples')
    axs[1].xtitle('Sampling')
    axs[1].legend()

    Pn = []
    for m in samples:
        Pn.append(Fnreal_vals[m]**2 + Fnimag_vals[m]**2)

    axs[2].plot(samples, Fnreal_vals, 'vb', label='Fn-Real')
    axs[2].plot(samples, Fnimag_vals, '^', label='Fn-Imag')
    axs[2]





