import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
import matplotlib as mpl
import random as rand

# mpl.matplotlib_fname()

# Poisson Distribution
def poisson(n, mean):
    return (mean**n * np.exp(-mean)) / factorial(n)

mean_vals = [1,5,10]

fig, axs = plt.subplots()
fig.suptitle(f'Poisson Distribution for $\\langle n \\rangle = 1, 5 ,10$')

for mean in mean_vals:
    n = np.arange(0, 20, 1)
    p = poisson(n, mean)
    axs.plot(n, p, label=f'$\\langle n \\rangle = {mean}$')

plt.legend() 
# plt.savefig('poisson.pdf')
plt.show()
plt.close()

N = 50
total_P = []
expectation_n = []
expectation_n_squared = []

for mean in mean_vals:
    total_P.append(0)
    expectation_n.append(0)
    expectation_n_squared.append(0)
    for n in range(N):
        P = poisson(n, mean)
        total_P[-1] += P
        expectation_n[-1] += n*P
        expectation_n_squared[-1] += n**2 * P
print(' Total P:', total_P,'\n',
      'Expectation value of n:', expectation_n,'\n', 'Expectation value of n^2:',expectation_n_squared) 

if np.isclose(total_P, 1).all():
    print('P(n) is normalized')
else:
    print('P(n) is not normalized')

# Variance
variance = []
for i in range(len(mean_vals)):
    variance.append(expectation_n_squared[i] - expectation_n[i]**2)
print('Variance:', variance)

# Standard Deviation
std_dev = np.sqrt(variance)
print('Standard Deviation:', std_dev,'\n','\n')



# Dart Throwing
def dart_thrower(L_):
    sector = rand.randrange(0, L_, 1)
    return sector

def dart_thrower_H_i(N_, rand_func_, L_):
    sectors = np.zeros(L_)
    dist = np.zeros(N_)
    for n in range(N_):
        sectors[rand_func_(L_)] += 1
    for n in range(N_):
        dist[n] = len(np.where(sectors == n)[0])
    return dist

def conc_exp(N_, rand_func_, Hn_i_, L_, T_):
    tot_dist = np.zeros(N_)
    for i in range(T_):
        distro = Hn_i_(N_, rand_func_, L_)
        tot_dist += distro
    return tot_dist

def mean_dist_n(N_, dist_):
    mean = 0
    for n in range(N_):
        mean += n*dist_[n]
    mean = mean/np.sum(dist_)
    return mean

def normalise_dist(dist_):
    return dist_/np.sum(dist_)
    


N = 50
L = [5, 100]
T = [100, 1000, 10000]



def plotting_and_analysis(T_, N_, L_):
    distro = []
    mean_of_dist = []
    normalised_distro = []

    for t in T_:
        distro.append(conc_exp(N_, dart_thrower, dart_thrower_H_i, L_, t))
        mean_of_dist.append(mean_dist_n(N_, distro[-1]))
        print(f'Mean of n, N = {N_}, T ={t}:', mean_of_dist[-1])
        normalised_distro.append(normalise_dist(distro[-1]))

        inds_sorted = np.argsort(normalised_distro[-1][np.where(normalised_distro[-1] > 0)])
        sorted_norm_distro = normalised_distro[-1][np.where(normalised_distro[-1] > 0)][inds_sorted]
        print('Smallest probabilities produced: ', sorted_norm_distro[:5])


    # Plotting
    fig, ax = plt.subplots(2,2)
    fig.set_size_inches(10,10)
    fig.suptitle(f'Distribution of n, {N_} darts, {L_} sectors')
    fig.tight_layout(pad=3.0)

    # Plotting non-normalised distribution on linear scale
    ax[0,0].grid()
    for t in range(len(T_)):
        ax[0,0].set_title(f'{T_} trials, Linear Scale')
        ax[0,0].plot(np.arange(N_), distro[t], linestyle='--')
        ax[0,0].scatter(np.arange(N_), distro[t], marker='o', label=f'T = {T[t]}')

    ax[0,0].legend()
    ax[0,0].xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax[0,0].set_xlabel('$n$')
    ax[0,0].set_ylabel('$H(n)$')

    # Plotting non-normalised distribution on log scale
    ax[0,1].grid()
    for t in range(len(T_)):
        ax[0,1].set_title(f'{T_} trials, Logarithmic Scale')
        ax[0,1].plot(np.arange(N_), distro[t], linestyle='--')
        ax[0,1].scatter(np.arange(N_), distro[t], marker='o', label=f'T = {T[t]}')

    ax[0,1].set_yscale('log')
    ax[0,1].legend()
    ax[0,1].xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax[0,1].set_xlabel('$n$')
    ax[0,1].set_ylabel('$H(n)$')

    # Plotting normalised distribution on linear scale
    ax[1,0].grid()
    for t in range(len(T_)):
        ax[1, 0].set_title(f'Normalised, {T_} trials, Linear Scale')
        ax[1, 0].plot(np.arange(N_), normalised_distro[t], linestyle='--', label=f'T = {T_[t]}')
        # ax[1, 0].scatter(np.arange(N_), normalised_distro[t], marker='o', label=f'Normalised Distribution, T = {T_[t]}')

    ax[1, 0].xaxis.set_minor_locator(plt.MultipleLocator(1)) 
    ax[1, 0].legend()
    ax[1, 0].set_xlabel('$n$')
    ax[1, 0].set_ylabel('$P(n)$')

    # Plotting normalised distribution on log scale
    ax[1,1].grid()
    for t in range(len(T_)):
        ax[1,1].set_title(f'Normalised, {T_} trials, Logarithmic Scale')
        ax[1,1].plot(np.arange(N_), normalised_distro[t], linestyle='--', label=f'T = {T_[t]}')
        # ax[1,1].scatter(np.arange(N_), normalised_distro[t], marker='o', label=f'Normalised Distribution, T = {T_[t]}')

    ax[1,1].xaxis.set_minor_locator(plt.MultipleLocator(1)) 
    ax[1,1].legend()
    ax[1,1].set_yscale('log')
    ax[1,1].set_xlabel('$n$')
    ax[1,1].set_ylabel('$P(n)$')

    # plt.savefig(f'/home/dj-lawton/Documents/Junior Sophister/Computer Simulation/Numerical Methods/Assignment 4/dart_throwingT{T_}N{N_}L{L_}.pdf')
    plt.show()
    plt.close()

for l in L:
    plotting_and_analysis(T, N, l)

print(' Name: David Lawton', '\n', 'Student ID: 22337087')