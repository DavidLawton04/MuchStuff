#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 19:23:21 2023

@author: dj-lawton
"""

import numpy as np
import matplotlib.pyplot as plt

A = 1090.0
b = 1.44
p = 0.033

x = np.arange(0.01, 1, 0.01)

def V(x): 
    return A*((np.e)**(-x/p))-(b/x)

#to find min/max of func, find roots of derivative
#Vdot = -F
def Vdot(x): 
    return (-A/p)*((np.e)**(-x/p))+(b/(x**2))

def Vdoubledot(x): 
    return (A*((np.e)**(-x/p)))/(p**2)-(2*(b/(x**3)))

x1 = 0.04


tol = [10**-10, 10**-9, 10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1]
roots=[]
Nsteps1=[]
equilofV=[]

def NRapprox(d1):
    
    for c in tol:
        nsteps1 = 0
        x1 = d1
        while abs(Vdot(x1)) > c:
            x1 = x1 - (Vdot(x1)/Vdoubledot(x1))
            nsteps1 += 1
        roots.append(x1)
        Nsteps1.append(nsteps1)
        equilofV.append(Vdot(x1))
    
    #The value of the force at the min. value of V(x) is 0, as it is the point where the attractive and repulsive forces are equal. 
    #from observation, there is no second root.
       

NRapprox(0.04)
print(roots)
print(Nsteps1)
print(equilofV)



font = {'fontname':'serif', 'size':12}
plt.figure()
plt.title('Root of V \'(x)', fontdict = font)
plt.plot(x, Vdot(x))
plt.plot(x, 0.0 * x)
plt.plot(roots[0], Vdot(roots[0]), 'b^')
plt.savefig('/home/dj-lawton/Documents/SF Lab Plots/Lab1 Plots/V\'(x)_graph_Ex3.pdf')
plt.show()
plt.close()


plt.figure()
plt.title('V(x)', fontdict = font)
plt.plot(x, V(x))
plt.savefig('/home/dj-lawton/Documents/SF Lab Plots/Lab1 Plots/V(x)_graph_Ex3.pdf')
plt.show()
plt.close()

plt.figure()
plt.title('V \'\'(x)', fontdict = font)
plt.plot(x, Vdoubledot(x))
plt.savefig('/home/dj-lawton/Documents/SF Lab Plots/Lab1 Plots/V\'\'(x)_graph_Ex3.pdf')
plt.show()
plt.close()