import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate

# Particle class definiton
class Particle():
    '''This defines particle object'''

    def __init__(self, mass, charge):
        self.mass = mass
        self.charge = charge
        self.__version = '1.0'


    def get_mass(self):
        print('Mass:', self.mass, 'kg')

    def get_charge(self):
        print('Charge:', (1.69E-19)*self.charge, 'C')
    
    def set_trajectory(self, field, y0, t_axis):

        def dydt(t, y):
            x,p = y
            q, m = self.charge*1.69E-19, self.mass
            dxdt = p/m
            dpdt = -q*field(x, t)
            return (dxdt, dpdt)
        
        (t0, tf) = t_axis[0], t_axis[-1]
        soln = integrate.solve_ivp(dydt, (t0, tf), y0, t_eval=t_axis)

        # Set the trajectory
        self.trajectory = (soln.y[0], soln.y[1], soln.t)

    def plot_trajectory(self):
        x, t = self.trajectory[0], self.trajectory[2]
        plt.plot(t, x)
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('Trajectory')
        plt.show()

    def _get_version(self):
        print(self.__version)


# Create Hadron subclass (inheritance)
class Hadron(Particle):
   
    # Overriding initialisation method
    def __init__(self, m, q, I):
        super().__init__(m, q)
        self.I = I

    