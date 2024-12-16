import numpy as np
from scipy.integrate import quad

def func1(x):
    return x**4* np.exp(x) /(np.exp(x) - 1)**2

def inner_integral(T):
    # Define Integral over x to be run for each T.
    result, _ = quad(func1, 0, 428/T, epsabs=1e-10)
    return result

def outer_integral(bounds=(0.5, 300)): # Bounds chosen since energy change for small T
                                       # is negligible, compared to 300K.
    # Define outside integral over T.
    def integrand(T):
        return T**3 *inner_integral(T)

    result, _ = quad(integrand, bounds[0], bounds[1], epsabs=1e-10)
    # Return final result including constants.
    return result*(9*8.3145/(428**3))

print(f'Energy required to raise temperature of aluminium from 0K to 300K is {outer_integral():.7}J',
      '\n', 'Name: David Lawton',
      '\n', 'Student No.: 22337087')
