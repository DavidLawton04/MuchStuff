# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:28:35 2023

@author: lawtond
"""

import numpy as np
import matplotlib.pyplot as plt

x=np.arange(-6.0, 12, 0.25)

a = -2
b = 12
c = -6

x1 = -2.0
x3 = 2.5
x5 = 7.0

def quadf(x): return a*(x**2) + b*x + c

y1 = quadf(x1)   
y3 = quadf(x3)
y5 = quadf(x5)

if y1 < 0 and y3 > 0 and y5 < 0:
    print ('values correctly initialised')

#Here I define my arrays and list used in the function.
nsteps1list = []
nsteps2list = []
tollist = [10**-10, 10**-9, 10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1]
root1list = []
y1list = []
root2list = []
y2list =[]

#Below I define the functions which return the required roots, lists + arrays.
def root1tol(a, b):
    x1 = a
    x3 = b
    x2 = 0.5 * (x1 + x3)
    
    
    for c in tollist:
        nsteps1 = 0
        while quadf(x2) != 0:
            nsteps1 += 1
           
            if quadf(x2) > 0:
                x3 = (x2)
                x2 = 0.5 * (x1 + x3)
            
            elif quadf(x2) < 0:
                x1 = x2
                x2 = 0.5 * (x1 + x3)
            
            if np.sqrt((quadf(x2))**2) < c:
                nsteps1list.append(nsteps1)
                root1list.append(x2)
                y1list.append(quadf(x2))
                break
        x1 = a
        x3 = b

def root2tol(a, b):
    x3 = a
    x5 = b
    x4 = 0.5 * (x3 + x5)
    nsteps2tol = 0
    for c in tollist:
        while quadf(x4) != 0:
            
            nsteps2tol += 1
            
            if quadf(x4) > 0:
                x3 = (x4)
                x4 = 0.5 * (x3 + x5)
            
            elif quadf(x4) < 0:
                x5 = x4
                x4 = 0.5 * (x3 + x5)
            
            if abs(quadf(x4)) < c:
                nsteps2list.append(nsteps2tol)
                root2list.append(x4)
                y2list.append(quadf(x4))
                break
        x3 = a
        x5 = b

root1tol(-2.0, 2.5)
print(root1list)
print(y1list)
print(tollist)
print(nsteps1list)
root1 = root1list[0]
print (root1)

root2 = root2tol(2.5, 7)
root2 = root2list[0]
print (root2)

#The below plot shows the quadratic function
plt.plot(x, a * x * x + b * x + c)
plt.plot(x, 0.0 * x)
plt.plot(root1, quadf(root1), 'go')
plt.plot(root2, quadf(root2), 'go')
plt.savefig('/home/dj-lawton/Documents/SF Lab Plots/Lab1 Plots/quadratic_graph_Ex1.pdf')
plt.show()

#The below plots show the relation between the accuracy and the no. of steps.
font = {'fontname':'serif', 'size':12}
plt.figure()
plt.ylim(0, 38)
plt.xlim(0, -10)
plt.title('Method 1', fontdict=font)
plt.xlabel("-Log10(tolerance)", fontdict=font)
plt.ylabel("Number of steps", fontdict=font)
plt.plot(np.log10(tollist), nsteps1list)
plt.plot(np.log10(tollist), nsteps1list, 'bo')
plt.savefig('/home/dj-lawton/Documents/SF Lab Plots/Lab1 Plots/nstepstolgraph.pdf')
plt.show()
plt.close()
