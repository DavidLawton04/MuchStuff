import numpy as np
import matplotlib.pyplot as plt

from particle import Particle, Hadron

proton = Particle(9.11E-31, -1)
proton.get_mass()
proton.get_charge()

def field(A, w, x, t):
    return A*np.cos(w*t)

# Fix params
x0, p0 = 0, 0
A, w = 1E-3, 1E3
t0, tf, dt = 0., 0.01, 1E-6
t_axis = np.arange(t0, tf, dt)

proton.set_trajectory(lambda x,t: field(A, w, x, t), (x0, p0), t_axis)
proton.plot_trajectory()
# print(proton._version)
proton._get_version()

pi_plus = Hadron(2.48E-24, 1, 1)
pi_plus.set_trajectory(lambda x,t: field(A, w, x, t), (x0, p0), t_axis)
pi_plus.plot_trajectory()
print(pi_plus.IsoSpin)
