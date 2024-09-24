#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 18:09:58 2023

@author: dj-lawton
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# V is velocity
# f(V) is the magnitude of the force resisting the velocity.
# b, c are factors depending on the medium
# d is the diameter of the projectile.

Kvals = np.arange(0.0009, 0.0011, 10**-7)
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

f1(1.6*(10**-4), 0.25)
print(f1vals)

plt.figure()
plt.xlabel('D * V')
plt.plot(Kvals, f1vals, 'b-', markersize=0.3)
plt.plot(Kvals, f2vals, 'y-', markersize=0.3,)
plt.legend(['b * V', 'c * V^2'], loc='upper left' )
# =============================================================================
# plt.savefig('/home/dj-lawton/Documents/SF Lab Plots/Lab3 Plots/linVSquad2.pdf')
# =============================================================================
plt.show()
plt.close()

#quad term negligible for D*V < ~0.0001
#both taken into account for ~0.0001 < D*V < ~0.005
#lin term negligible for D*V > ~0.005

#Baseball Case
D = 0.07
V = 5
print(D*V)
#optimal case for baseball: take only quad term, lin neg.

#Oil droplet case
D = 1.5*(10**-6)
V = 5*(10**-5)
print(D*V)
#optimal case for oil droplet: take only linear term, quad neg.

#Raindrop case
D = 1*(10**-3)
V = 1
print(D*V)
#optimal case for raindrop: take both terms into account, neither neg.