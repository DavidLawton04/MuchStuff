import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('/home/dj-lawton/Documents/Junior Sophister/Learning_C++/convergence.csv')
print(data)

fig, ax = plt.subplots()
fig.set_size_inches(20,5)
fig.suptitle('Convergence of Metropolis Algorithm to Thermodynamic Limit')
ax.hlines(200, 0, 15000, color='red', linestyle='--')
ax.plot(data['index'], -data['mean energies'])
ax.set_yscale('log')
ax.set_xlim(0,4000)
# ax.set_xscale('log')
plt.savefig('/home/dj-lawton/Documents/Junior Sophister/Learning_C++/convergence.png')
plt.show()
plt.close()

data2 = pd.read_csv('/home/dj-lawton/Documents/Junior Sophister/Learning_C++/values_for_T1.0_hvary.csv')
fig, ax = plt.subplots(2,2)
fig.set_size_inches(10,10)
fig.suptitle('Thermodynamic Quantities varying w.r.t. $h$')

ax[0,0].plot(data2['h'],data2['Energy'])
ax[0,1].plot(data2['h'],data2['Magnetisation'])
ax[1,0].plot(data2['h'],data2['Specific Heat'])
ax[1,1].plot(data2['h'],data2['Magnetic Susceptibility'])
plt.savefig('/home/dj-lawton/Documents/Junior Sophister/Learning_C++/constT1.0varyingh.png')
plt.show()
plt.close()