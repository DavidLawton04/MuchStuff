#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 09:49:39 2023

@author: dj-lawton
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def f(x):
    g = np.exp(x)
    return g


def F(a, b, n):
    h = (b-a)/(n)
    a_b = np.arange(a, b, (b-a)/(n))
    J1 = 0
    J2 = 0
    for j in range(1, len(a_b)):
        if j % 2 == 0:
            J1 += f(a_b[j])
        elif j % 2 != 0:
            J2 += f(a_b[j])

    F_ = (h/3) * (f(a) + 2*J1 + 4*J2 + f(b))
    return F_
    print(F_)


values = []
for n in np.arange(1, 400):
    values.append(F(0, 1, n))
    print(f'Integral value: {values[n-1]}\
          Expected value: 1.718281828459045\
          Step Number: {n}')
F(0, 1, 4)

plt.figure()
plt.plot(np.array(range(1, 400)), values, '.b', markersize=5)
plt.xlabel('Step No.')
plt.savefig('Integral Accuracy2.pdf')
