import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy import special, interpolate, integrate, optimize

# SPECIAL FUNCTIONS

r = np.linspace(0, 100, 1000)
plt.plot(r, special.jn(1, r), label='J0')   
# plt.show()
plt.close()

# Numerical integration
# simpson's rule for samples of f(x) at {x0, x1, x2, x3,...}
# romberg integration for samples of f(x) at {x0, x1, x2, x3,...}, 2^k +1 points, evenly spaced
# quad integration for samples of f(x) at {x0, x1, x2, x3,...}, adaptive quadrature


# Define function f(x)
def f(x):
    return np.sin(x**2)**2*np.exp(-np.abs(x))


N = 2**10 + 1
x_min, x_max = 0, 6
x = np.linspace(x_min, x_max, N)

# Plot f(x)
plt.plot(x, f(x))
# plt.show()
plt.close()

# Numerical integration
# simpson's rule
t_start = time()
result = integrate.simpson(f(x), x)
t_end = time()
tot_time = t_end - t_start
print('Simpson\'s rule:  %.16f in %.16f seconds' % (result, tot_time))

# romberg integration
t_start = time()
dx = x[1] - x[0]
result = integrate.romb(f(x), dx)
t_end = time()
tot_time = t_end - t_start
print('Romberg integration:  %.16f in %.16f seconds' % (result, tot_time))


# quad integration
t_start = time()
result, err = integrate.quad(f, x_min, x_max)
t_end = time()
tot_time = t_end - t_start
print('Quad integration:  %.16f in %.16f seconds, with error %.16f' % (result, tot_time, err))

result, err = integrate.quad(f, 0, np.inf, limit=100)
print(result, err)

# dblquad, tplquad, nquad, integrate over multiple variables
# limits can be functions of the other variables


def f2(y, x):
    return x*y**2

def h(x):
    return x

def g(x):
    return 0

x_min, x_max = 0, 1
y_min, y_max = 0, 1

result1, err1 = integrate.dblquad(f2, x_min, x_max, g, h)

result, err = integrate.dblquad(f2, x_min, x_max, 0, lambda x: x)

print(result1, err1)
print(result, err) 

# INTERPOLATION

#Fine grid
x_min, x_max, dx = 0, 6, 0.005
x1 = np.arange(x_min, x_max, dx)

#Coarse grid
x2 = np.arange(x_min, x_max, 0.2)

#Create interpolating function
f_interp = interpolate.interp1d(x2, f(x2), kind='cubic', fill_value='extrapolate')
print(f_interp)

plt.figure()
plt.plot(x1, f(x1), label='f(x), fine grid')
plt.plot(x2, f(x2), 'o', label='f(x), coarse grid')
plt.plot(x1, f_interp(x1), 'r--', label='Interpolated f(x)')
plt.text(3, 0.05, 'aliasing of extrapolation', fontsize=10, color='red')
plt.legend()
# plt.show()
plt.close()

# OPTIMIZATION
f_min = optimize.minimize(lambda x: -f(x), 1)

print(f_min.x, f_min.fun)

# works for multivariate functions with different variables as entries of an array.

# multidim eg
def f3(x, y):
    return np.sinc(y-1)*np.exp(-x**2)

f3_min = optimize.minimize(lambda X: -f3(X[0], X[1]), np.array([0,0]))
print(f3_min.x, f3_min.fun) # effectively (0,1)