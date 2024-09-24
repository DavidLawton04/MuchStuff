  # -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:49:40 2023

@author: lawtond
"""

import numpy as np
import matplotlib.pylab as plt


x=np.arange(-6.0, 12, 0.25)

a = -2
b = 12
c = -6

def quadf(x): return a*(x**2) + b*x + c
def quadfderiv(x): return 2 * a * x + b


x1 = -2
x2 = 7

def init(e1, e2):
    if quadf(e1)*quadf(e2)>0:
        print("values correctly initialised")
    else:
        print('pick new values')
        

tol =[10**-10, 10**-9, 10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1]
roots1=[]
Nsteps1=[]
Nsteps2=[]
yvalues=[]
roots2=[]
def NRapprox(d1, d2):
    
    for c in tol:
        nsteps1 = 0
        nsteps2 = 0
        x1 = d1
        while abs(quadf(x1)) > c:
            x1 = x1 - (quadf(x1)/quadfderiv(x1))
            nsteps1 += 1
        roots1.append(x1)
        Nsteps1.append(nsteps1)
        yvalues.append(quadf(x1))
        #secondroot approximation starts on other side of maxima
        x2 = d2
        while abs(quadf(x2)) > c:
            x2 = x2 - (quadf(x2)/quadfderiv(x2))
            nsteps2 += 1
        roots2.append(x2)
        Nsteps2.append(nsteps2)

init(x1, x2)
NRapprox(-2, 7)
print(roots1)
print(Nsteps1)
print(yvalues)
#The below plot shows the quadratic function
plt.figure()
plt.plot(x, quadf(x))
plt.plot(x, 0.0 * x)
plt.plot(roots1[0], quadf(roots1[0]), 'b^')
plt.plot(roots1[1], quadf(roots1[1]), 'b^')
plt.plot(x, quadfderiv(x))
plt.savefig('/home/dj-lawton/Documents/SF Lab Plots/Lab1 Plots/quadratic_graph_Ex2.pdf')
plt.show()
plt.close()


font = {'fontname':'serif', 'size':12}
plt.figure()
plt.ylim(0, 38)
plt.xlim(0, -10)
plt.title('Method 2', fontdict=font)
plt.xlabel('Log10(tolerance)', fontdict=font)
plt.ylabel('Number of steps', fontdict=font)
plt.plot(np.log10(tol), Nsteps1)
plt.plot(np.log10(tol), Nsteps1, 'bo')
plt.savefig('/home/dj-lawton/Documents/SF Lab Plots/Lab1 Plots/Tol_vs_steps_Ex2.pdf')
plt.show()
plt.close()
