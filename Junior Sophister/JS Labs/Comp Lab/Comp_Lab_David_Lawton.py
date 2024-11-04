import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

# Pick some number of evenly spaced points N
N = 1000
Psi = np.zeros(N)

# Set gamma squared to some value, physical params contained in this
GammaSquare = 200

# Define the dimensionless potential
def nu(xtilde):
    return -1

# Pick some trial dimensionless energy, nrg, which must be greater
# than the minimum of the dimensionless potential, nu
# (see Griffiths Quantum mechanics Problem 2.2)
nrg = -0.9

# Set array of dimensionless x values
xtilde_vals = np.linspace(0, 1, N)


# Define iterative method for solving the Schrodinger equation
def iterative_method(Psi_, k_squared_vals_, N_):
    l_squared = (1/N)**2
    p_vals, q_vals = 1 - (5/12) * l_squared * k_squared_vals_, 1 + (1/12) * l_squared * k_squared_vals_
    for i in range(2, N_):
        Psi_[i] = (2*p_vals[i-1]*Psi_[i-1] - q_vals[i-2]*Psi_[i-2])/q_vals[i]
    return Psi_

def analysis(Psi_, nrg_, nu_, xtilde_vals_, N_, GammaSquare_):

    # Set the dimensionless energy array so that calculation is easier.
    nrg_vals = nrg_*np.ones(N_)

    # Calculate the potential and k squared values from the inputted potential.
    # points and energy.
    potential_vals = [nu_(x) for x in xtilde_vals_]
    k_squared_vals = GammaSquare_ * (nrg_vals - potential_vals)

    # For nu=-1, first two points given by
    Psi_[0], Psi_[1] = 0, 1E-4

    # Calculate the wavefunction from these points.
    Psi_ = iterative_method(Psi_, k_squared_vals, N_)

    fig, ax = plt.subplots()
    fig.suptitle('Numerical Solution to the Schrodinger Equation')

    ax.grid(alpha=0.6)
    ax.plot(xtilde_vals_, Psi_, label='Numerical Solution')
    ax.set_xlabel('$x/L$')
    ax.set_ylabel('$\psi(x/L)$')

    plt.show()
    plt.close()

# analysis(Psi, nrg, nu, xtilde_vals, N, GammaSquare)


def analysis_no_plot(Psi_, nrg_, nu_, xtilde_vals_, N_, GammaSquare_):

    # Set the dimensionless energy array so that calculation is easier.
    nrg_vals = nrg_*np.ones(N_)

    # Calculate the potential and k squared values from the inputted potential.
    # points and energy.
    potential_vals = [nu_(x) for x in xtilde_vals_]
    k_squared_vals = GammaSquare_ * (nrg_vals - potential_vals)

    # For nu=-1, first two points given by
    Psi_[0], Psi_[1] = 0, 1E-4

    # Calculate the wavefunction from these points.
    Psi_ = iterative_method(Psi_, k_squared_vals, N_)
    return Psi_

def Shooting_method(nrg_, nu_, xtilde_vals_, N_, GammaSquare_):
        dnrg = 1E-4
        tolerance = 1E-14

        Psi_ = np.zeros(N_)
        Psi_ = analysis_no_plot(Psi_, nrg_, nu_, xtilde_vals_, N_, GammaSquare_)
        Psi_end_vals = [Psi_[-1]]

        while dnrg > tolerance:
            nrg_ += dnrg
            Psi_prime = analysis_no_plot(Psi_, nrg_, nu_, xtilde_vals_, N_, GammaSquare_)
            Psi_end_vals.append(Psi_prime[-1])
            if Psi_end_vals[-1]*Psi_end_vals[-2] < 0:
                dnrg = -dnrg/2
        return nrg_

energy = Shooting_method(nrg, nu, xtilde_vals, N, GammaSquare)
print(energy)
    
        
