#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt


def dft(N, func_vals):
    fn_real = []
    fn_imag = []
    fn = []
    for n in range(0, N):
        real_val = imag_val = 0
        for m in range(0, N):
            real_val += func_vals[m] * np.cos(2 * np.pi * m * n / N)
            imag_val += -func_vals[m] * np.sin(2 * np.pi * m * n / N)
        fn_real.append(real_val)
        fn_imag.append(imag_val)
        fn.append(complex(real_val, imag_val))
    return fn, fn_real, fn_imag


def DFTFunc_n(N, nlist, m, func_vals):
    Fnn_real = []
    Fnn_imag = []
    for n in nlist:
        Fnn_real.append(func_vals[m] * np.cos(2 * np.pi * m * n / N))
        Fnn_imag.append(-func_vals[m] * np.sin(2 * np.pi * m * n / N))
    return Fnn_real, Fnn_imag


def back_transform(N, fn_real, fn_imag):
    fm_real = []
    fm_imag = []
    for m in range(0, N):
        value = 0
        for n in range(0, N):
            fn = complex(fn_real[n], fn_imag[n])
            theta = 2 * np.pi * m * n / N
            value += fn * complex(np.cos(theta), np.sin(theta))
        fm_real.append(np.real(value) / N)
        fm_imag.append(np.imag(value) / N)
    return fm_real, fm_imag


def analyse(func, N, h, funcstr):
    tau = N * h

    print(f"\nDFT, N={N} h={h}")
    print(f"Sample frequency: {1/h} Hz")

    w1 = 2 * np.pi / tau
    print(f"Fundamental frequency {w1=}")
    wN = (1 / (2 * h)) - (1 / (N * h))
    print(f"Nyquist frequency {wN=}")

    # Ideal sampling interval
    freq = 6 * np.pi / (2 * np.pi)
    print(f"Actual frequency: {freq} Hz")
    period = 1 / freq
    print(f"Actual period: {period} s")
    # Tau equals period of function = N * h.
    h_ideal = period / N
    print(f"Ideal sampling interval: {h_ideal}")
    print(f"Ideal sampling frequency: {1/h_ideal} Hz")

    actual_func_vals = []
    timesteps = np.arange(0, N * h, 0.01)
    for t in timesteps:
        actual_func_vals.append(func(t))

    func_vals = []
    sample_times = []
    for m in range(0, N):
        sample_times.append(m * h)
    for t in sample_times:
        func_vals.append(func(t))
    fn, fn_real, fn_imag = dft(N, func_vals)

    samples = range(0, N)

    fig, axs = plt.subplots(2, 2)
    fig.set_figwidth(18)
    fig.set_figheight(10)
    fig.suptitle(f"DFT, func={funcstr} N={N} h={h}")

    axs[0, 0].plot(timesteps, actual_func_vals, label='function')
    axs[0, 0].legend()
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_title("Actual function")

    axs[0, 1].plot(samples, func_vals, drawstyle='steps-pre', label='function')
    axs[0, 1].plot(samples, func_vals, '.', label='function samples')
    axs[0, 1].legend()
    axs[0, 1].set_xlabel('Sample')
    axs[0, 1].set_title("Sampled function")

    # (10) Power spectrum
    pn = []
    for n in range(0, N):
        pn.append(fn_real[n] ** 2 + fn_imag[n] ** 2)

    axs[1, 0].plot(samples, fn_real, '^', label='Fn Real')
    axs[1, 0].plot(samples, fn_imag, 'v', label='Fn Imaginary')
# =============================================================================
#     axs[1, 0].plot(samples, pn, '^', label='Power')
# =============================================================================
    axs[1, 0].legend()
    axs[1, 0].set_xlabel('Sample')
    axs[1, 0].set_title("Fourier (Real, Imaginary), Power Spectrum")

    # (10) Back transform
    fm_real, fm_imag = back_transform(N, fn_real, fn_imag)

    axs[1, 1].plot(samples, func_vals, drawstyle='steps-pre', label='function')
    axs[1, 1].plot(samples, func_vals, 'o', label='function samples')
    axs[1, 1].plot(samples, fm_real, '.', label='fm_real')
    axs[1, 1].plot(samples, fm_imag, 'v', label='fm_imag')
    axs[1, 1].legend()
    axs[1, 1].set_xlabel('Sample')
    axs[1, 1].set_title("Function vs back tranform real, imag")

    plt.savefig(f"{funcstr} {N} {h}.pdf")
    plt.show()
    plt.close()

    m = 6
    Fnn_real, Fnn_imag = DFTFunc_n(N, samples, m, func_vals)

    fig, axs = plt.subplots(3)
    fig.set_figwidth(18)
    fig.set_figheight(10)
    fig.suptitle(f"Fn, Pn as function of n")

    Pn = []
    axs[0].plot(samples, Fnn_real, '--')
    axs[1].plot(samples, Fnn_imag, '--')

    Pn = []
    for n in samples:
        Pn.append(Fnn_real[n]**2 + Fnn_imag[n]**2)

    axs[2].plot(samples, Pn, '.')
    plt.show()


# 3.(1)-(8) f(t) = sin(0.45 * pi * t)
def func1(t):
    return np.sin(0.45 * np.pi * t)


N = 128  # Number of samples.
h = 0.1  # Sampling interval (s).
analyse(func1, N, h, "sin(0.45_pi_t)")

# (9)(10) f(t) = cos(6 * pi * t)
# (11) f(t) = cos(6 * pi * t)


def func2(t):
    return np.cos(6.0 * np.pi * t)


N = 32
for h in (0.6, 0.2, 0.1, 0.04, 0.01):
    analyse(func2, N, h, "cos_6_pi_t")


# (Supplementary (1) - (4)) f(t) = cos(3 * t)


def func3(t):
    return np.cos(3.0 * t)


h = 1
for N in (8, 16, 32, 64):
    analyse(func3, N, h, "cos_3_t")


# (Supplementary (5) - (8)) f(t) = e^-kt * sin(t)
def func4(t):
    k = 0.2
    return np.sin(t) * math.exp(-k * t)


N = 20
h = 0.5
analyse(func4, N, h, "e-kt_sin_t")


# (Supplementary (9) - (10)) f(t) = 1, 0<t<tau, 0, t<0, t>tau
def func5(t):
    tau = 1
    if t < 0 or t > tau:
        return 0
    else:
        return 1


N = 20
h = 0.5
for N, h in ((10, 1), (20, 0.5), (40, 0.25), (80, 0.125)):
    analyse(func5, N, h, "pulse")
