import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df_pop1 = pd.read_csv('Computer Simulation/Numerical Methods/Assignment 2/population-gapminder(3).dat', skiprows=1, names=['Year', 'Population'], delim_whitespace=True)
df_pop2 = pd.read_csv('Computer Simulation/Numerical Methods/Assignment 2/population-census-bureau(1).dat', skiprows=1, names=['Year', 'Population'], delim_whitespace=True)
# df_pop3 = pd.concat((df_pop2,df_pop1), axis=0)
print(df_pop2.head(),'\n','\n', df_pop1.head())

df_pop1['Ln Population'] = np.log(df_pop1['Population'])
df_pop2['Ln Population'] = np.log(df_pop2['Population'])

def logaritmic_growth(x, a, b):
    return a + b*x

df_pop1['subset_pop1'] = df_pop1['Ln Population'].iloc[0:3]
df_pop2['subset_pop2'] = df_pop2['Ln Population'].iloc[0:45]
params1, covariance1 = curve_fit(logaritmic_growth, df_pop1['Year'], df_pop1['subset_pop1'])
params2, covariance2 = curve_fit(logaritmic_growth, df_pop2['Year'], df_pop2['subset_pop2'])
sigma1 = np.sqrt(np.diag(covariance1))
sigma2 = np.sqrt(np.diag(covariance2))

fig, axs = plt.subplots(1,2)
fig.suptitle('Population vs Year')
fig.set_figwidth(10)
fig.set_figheight(5)


axs[0].plot(df_pop1['Year'], df_pop1['Population'], label='Gapminder')
axs[0].plot(df_pop2['Year'], df_pop2['Population'], label='Census Bureau')
axs[0].fill_between(df_pop1['Year'], df_pop1['Population'], color='blue', alpha=0.3)
axs[0].fill_between(df_pop2['Year'], df_pop2['Population'], color='red', alpha=0.3) 
axs[0].set_xlabel('Year')
axs[0].set_ylabel('N')
axs[0].set_title('Population vs Year')
axs[0].legend()


axs[1].set_xlabel('Year')
axs[1].set_ylabel('$\ln(N)$')
axs[1].set_title('Ln Population vs Year')
axs[1].plot(df_pop1['Year'], df_pop1['Ln Population'], label='Gapminder Dataset')
axs[1].plot(df_pop2['Year'], df_pop2['Ln Population'], label='Census Bureau Dataset')
axs[1].plot(df_pop1['Year'].iloc[0:3], logaritmic_growth(df_pop1['Year'].iloc[0:3], *params1), label='Gapminder Fit', color='blue')
axs[1].plot(df_pop2['Year'].iloc[0:45], logaritmic_growth(df_pop2['Year'].iloc[0:45], *params2), label='Census Bureau Fit', color='red')
axs[1].fill_between(logaritmic_growth(df_pop1['Year'].iloc[0:3], params1[0]+sigma1[0,0], params1[1, 1]+sigma1[1,1], color='blue', alpha=0.3), logaritmic_growth(df_pop1['Year'].iloc[0:3], params1[0]-sigma1[0,0], params1[1, 1]-sigma1[1,1]), color='blue', alpha=0.3)
axs[1].legend()

plt.savefig('Computer Simulation/Numerical Methods/Assignment 2/population_growth.png')
plt.show()