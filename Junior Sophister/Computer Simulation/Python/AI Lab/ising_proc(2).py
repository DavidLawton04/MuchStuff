import numpy as np
import matplotlib.pyplot as plt
from time import time

start_time = time()


# Initialise the system
L, T, h = 100, 1.0, 0.1
state = np.ones(L)

# Function to find the total magnetisation
def get_mag(state):
    return sum(state)

# Function to find the total energy
def get_energy(state, h):
    L = len(state)
    E = -h*get_mag(state)
    for site in range(L):
        E -= state[site-1]*state[site]
    return E

# Local field acting on a given site
def local_field(state, site, h):
    L = len(state)
    return h + state[site-1] + state[(site+1)%L]

# Sweep through the lattice, randomly updating state
def sweep(state, T, h):
    
    L = len(state)
    for site in range(L):
        # Local spin state and energy change of a spin flip
        s = state[site]
        dE = 2*s*local_field(state, site, h)
        
        # Decide whether to flip
        acc = np.exp(-dE/T)
        r = np.random.rand()
        if r < acc:
            # Update the local spin state 
            state[site] = -s
    # Numpy arrays are mutable! No need for a return value

# Full metropolis algorithm
def metropolis(state, T, h, n_sweeps = 1000, init_sweeps = 200):
    
    # Initialise the lattice
    for nn in range(init_sweeps):
        sweep(state, T, h)
    
    # Run sweeps and store observables at each step
    M, M2, E, E2 = 0.0, 0.0, 0.0, 0.0
    for nn in range(n_sweeps):
        sweep(state, T, h)
        m, e = get_mag(state), get_energy(state, h)
        M += m
        E += e
        M2 += m**2
        E2 += e**2
    
    # Average and return the results
    M, E = M/n_sweeps, E/n_sweeps
    M2, E2 = M2/n_sweeps, E2/n_sweeps
    dM, dE = (M2-M**2)**0.5, (E2 - E**2)**0.5
    return M, dM, E, dE

# Compute the magnetisation as a function of magnetic field
# for fixed temperature
L, T = 100, 2.0
h_list = np.linspace(0.0, 4.0, 20)
M_list, dM_list = [], []
for h in h_list:
    state = np.random.choice([-1,1], L)
    M, dM, E, dE = metropolis(state, T, h)
    M_list.append(M/L)
    # dM_list.append(dM)
    
# Plot the results
plt.figure()
plt.plot(h_list, M_list)
# plt.plot(h_list, dM_list)
plt.xlabel('h/J')
plt.ylabel('magnetisation density')
plt.legend(['$\langle M\\rangle$', '$\\delta M$'])


# Compute the magnetisation density as a function of temperature at zero field
L, h = 100, 0.0
T_list = np.linspace(0.01, 2.0, 20)
M_list, dM_list = [], []
state = np.random.choice([-1,1], L)
M, dM, E, dE = metropolis(state, T_list[0], h)
M_list.append(M/L)
for T in T_list[1:]:
    M, dM, E, dE = metropolis(state, T, h)
    M_list.append(M/L)
    # dM_list.append(dM)
    
# Plot the results
plt.figure()
plt.plot(T_list, M_list)
plt.xlabel('T/J')
plt.ylabel('magnetisation density')
plt.title('h=%.1f' %(h))
plt.legend(['$\langle M\\rangle$'])
plt.ylim([-1,1])

# Compute the magnetisation as a function of temperature in fixed field
L, h = 100, 1.0
T_list = np.logspace(-1.0, 2.0, 20)
M_list = []

# Do one iteration
state = np.random.choice([-1,1], L)
M, dM, E, dE = metropolis(state, T_list[0], h)
M_list.append(M/L)
for T in T_list[1:]:
    M, dM, E, dE = metropolis(state, T, h, init_sweeps = 0)
    M_list.append(M/L)

# Plot the results
plt.figure()
plt.plot(T_list, M_list)
plt.xlabel('k_B T/J')
plt.ylabel('magnetisation density')
plt.title('h=%.1f' %(h))
plt.yscale('log')
plt.xscale('log')




end_time = time()

print('Total time =',end_time-start_time,'s')
