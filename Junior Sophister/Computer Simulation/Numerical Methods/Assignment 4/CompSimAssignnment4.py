import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.matplotlib_fname()

# Poisson Distribution
def poisson(n, mean):
    return (mean**n * np.exp(-mean)) / factorial(n)

mean_vals = [1,5,10]

fig, axs = plt.subplots()
fig.suptitle(f'Poisson Distribution for $\langle n \\rangle = 1, 5 ,10$')

for mean in mean_vals:
    n = np.arange(0, 20, 1)
    print(n)
    p = poisson(n, mean)
    axs.plot(n, p, label=f'$\langle n \\rangle = {mean}$')

plt.legend() 
plt.savefig('poisson.pdf')
plt.show()

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
print('Standard Deviation:', std_dev)