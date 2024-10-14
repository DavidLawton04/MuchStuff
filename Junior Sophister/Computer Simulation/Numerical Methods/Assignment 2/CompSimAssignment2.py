import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Change to your own path for this data
df_pop1 = pd.read_csv('Computer Simulation/Numerical Methods/Assignment 2/population-gapminder(3).dat', skiprows=1, names=['Year', 'Population'], delim_whitespace=True)
df_pop2 = pd.read_csv('Computer Simulation/Numerical Methods/Assignment 2/population-census-bureau(1).dat', skiprows=1, names=['Year', 'Population'], delim_whitespace=True)

df_pop1['Ln Population'] = np.log(df_pop1['Population'])
df_pop2['Ln Population'] = np.log(df_pop2['Population'])

# Linear function for fitting Logarithmic data to
def linear_func(x, a, b):
    return a + b*x

subset_df_pop1 = df_pop1.iloc[0:4]
# subset_df_pop2 = df_pop1.iloc[5:10]
subset_df_pop3 = df_pop2.iloc[0:45]

# Fitting the logarithm of the population to the linear function
params1, covariance1 = curve_fit(linear_func, subset_df_pop1['Year'], subset_df_pop1['Ln Population'])
# params2, covariance2 = curve_fit(linear_func, subset_df_pop2['Year'], subset_df_pop2['Ln Population'])
params3, covariance3 = curve_fit(linear_func, subset_df_pop3['Year'], subset_df_pop3['Ln Population'])

# Standard deviation of the parameters
sigma1 = np.sqrt(np.diag(covariance1))
# sigma2 = np.sqrt(np.diag(covariance2))
sigma3 = np.sqrt(np.diag(covariance3))


fig, axs = plt.subplots(1,2)
fig.suptitle('Population vs Year')
fig.set_figwidth(10)
fig.set_figheight(5)

# Plotting original data
axs[0].plot(df_pop1['Year'], df_pop1['Population'], label='Gapminder')
axs[0].plot(df_pop2['Year'], df_pop2['Population'], label='Census Bureau')
axs[0].fill_between(df_pop1['Year'], df_pop1['Population'], color='blue', alpha=0.3)
axs[0].fill_between(df_pop2['Year'], df_pop2['Population'], color='red', alpha=0.3) 
axs[0].set_xlabel('Year')
axs[0].set_ylabel('$N\cdot 10^{-6}$')
axs[0].set_title('Population vs Year')

# Plotting the logarithm of the population vs year
axs[1].set_xlabel('Year')
axs[1].set_ylabel('$\ln(N\cdot 10^{-6})$')
axs[1].set_title('Ln Population vs Year')
axs[1].plot(df_pop1['Year'], df_pop1['Ln Population'], label='Gapminder Dataset')
axs[1].plot(df_pop2['Year'], df_pop2['Ln Population'], label='Census Bureau Dataset')

# Plot the fit
axs[1].plot(subset_df_pop1['Year'], linear_func(subset_df_pop1['Year'], *params1), label='Gapminder Fit', color='blue')
# axs[1].plot(subset_df_pop2['Year'], linear_func(subset_df_pop2['Year'], *params2), label='Gapminder Fit 2', color='red')
axs[1].plot(subset_df_pop3['Year'], linear_func(subset_df_pop3['Year'], *params3), label='Census Bureau Fit', color='green')

# ~65% confidence interval
axs[1].fill_between(subset_df_pop1['Year'],linear_func(subset_df_pop1['Year'], params1[0]+1*sigma1[0], params1[1]), linear_func(subset_df_pop1['Year'], params1[0]-1*sigma1[0], params1[1]), color='blue', alpha=0.3)
# axs[1].fill_between(subset_df_pop2['Year'],linear_func(subset_df_pop2['Year'], params2[0]+1*sigma2[0], params2[1]), linear_func(subset_df_pop2['Year'], params2[0]-1*sigma2[0], params2[1]), color='red', alpha=0.3)
axs[1].fill_between(subset_df_pop3['Year'],linear_func(subset_df_pop3['Year'], params3[0]+1*sigma3[0], params3[1]), linear_func(subset_df_pop3['Year'], params3[0]-1*sigma3[0], params3[1]), color='green', alpha=0.3)



print(f'Slope of first fit: {params1[1] } ± {sigma1[1]}', f'Intercept of first fit: {params1[0]} ± {sigma1[0]}')
# print(f'Slope of second fit: {params2[1]} ± {sigma2[1]}', f'Intercept of second fit: {params2[0]} ± {sigma2[0]}')
print(f'Slope of third fit: {params3[1]} ± {sigma3[1]}', f'Intercept of third fit: {params3[0]} ± {sigma3[0]}')

# Calculating n_0 and lambda for the subsets
lambdas = params1[1], params3[1]
sigma_lambdas = sigma1[1], sigma3[1]
n_0 = np.exp(params1[0]+lambdas[0]*1750), np.exp(params3[0]+lambdas[1]*1950)
sigma_n_0s = np.sqrt((np.exp(params1[0]+lambdas[0]*1750)*(sigma1[0]))**2 + (np.exp(params1[0]+lambdas[0]*1750)*(1750*sigma_lambdas[0]))**2), np.sqrt((np.exp(params3[0]+lambdas[1]*1950)*(sigma3[0]))**2 + (np.exp(params3[0]+lambdas[1]*1950)*(1950*sigma_lambdas[1]))**2)

print(f'Paste these inside a five column latex tabular environment to view more clearly')
print(f'Subset i & $a_i$ & $b_i$ & $\sigma_{{a_i}}$ & $\sigma_{{b_i}}$ \\\\','\n',f'Subset 1 & {params1[0]:.4} & {params1[1]:.4} & {sigma1[0]:.4} & {sigma1[1]:.4} \\\\','\n', f'Subset 2 & {params3[0]:.4} & {params3[1]:.4} & {sigma3[0]:.4} & {sigma3[1]:.4} \\\\')
print(f'Subset i & $n_0$ & $\lambda$ & $\sigma_{{n_0}}$ & $\sigma_{{\lambda}}$ \\\\','\n',f'Subset 1 & {n_0[0]:.4} & {lambdas[0]:.4} & {sigma_n_0s[0]:.4} & {sigma_lambdas[0]:.4} \\\\','\n', f'Subset 2 & {n_0[1]:.4} & {lambdas[1]:.4} & {sigma_n_0s[1]:.4} & {sigma_lambdas[1]:.4} \\\\')

# Plotting the fit back onto the original data
time_range1 = np.linspace(1750, 1940, 100)
time_range2 = np.linspace(1950, 2016, 100)
axs[0].plot(time_range1, np.exp(params1[0] + lambdas[0]*time_range1), color='purple', linestyle='--', label='Gapminder Fit')
axs[0].plot(time_range2, np.exp(params3[0] + lambdas[1]*time_range2), color='green', linestyle='--', label='Census Bureau Fit')
axs[0].fill_between(time_range1, np.exp((params1[0] + sigma1[0]) + (lambdas[0] + sigma_lambdas[0])*time_range1), np.exp((params1[0] - sigma1[0]) + (lambdas[0] - sigma_lambdas[0])*time_range1), color='purple', alpha=0.3)
axs[0].fill_between(time_range2, np.exp((params3[0] + sigma3[0]) + (lambdas[1] + sigma_lambdas[1])*time_range2), np.exp((params3[0] - sigma3[0]) + (lambdas[1] - sigma_lambdas[1])*time_range2), color='green', alpha=0.3)


#Legend
axs[0].legend()
axs[1].legend()

# plt.savefig('Computer Simulation/Numerical Methods/Assignment 2/population_growth.pdf')
plt.show()
plt.close()

fig, axs = plt.subplots()
fig.suptitle('Non-constant growth rate')
axs.set_xlabel('Year')
axs.axvline(2030, color='black', linestyle=':')
axs.text(2030, -1000, '2030')
axs.set_ylabel('Population (Millions)')
axs.plot(df_pop1['Year'], df_pop1['Population'], label='Gapminder Data')
axs.plot(df_pop2['Year'], df_pop2['Population'], label='US Census Bureau Data')
axs.fill_between(df_pop1['Year'], df_pop1['Population'], color='blue', alpha=0.3)
axs.fill_between(df_pop2['Year'], df_pop2['Population'], color='red', alpha=0.3)


means = [subset_df_pop1['Year'].mean(), subset_df_pop3['Year'].mean()]

# Fitting the growth rate to a linear function
lambdaparams, lambdacovariance = curve_fit(linear_func, means, lambdas, sigma=sigma_lambdas, maxfev=10000)

print(f'The population in 2022 is predicted to be {df_pop2["Population"].loc[66]*np.exp(linear_func(2022, *lambdaparams)*(2022-2016)):.4} million, the actual population was 7.951 * 10^3 million')

# Error not found due to low number of data points.
sigma_lambda_fit = np.sqrt(np.diag(lambdacovariance))

# Establish the range over which predictions will be made
prediction_range = np.linspace(2016, 2030, 30)

# Plotting the prediction
axs.plot(prediction_range, df_pop2['Population'].loc[66]*np.exp(linear_func(prediction_range, *lambdaparams)*(prediction_range-2016)), color='purple', linestyle='--', label='Prediction (Lambda Fit)')
axs.fill_between(prediction_range, df_pop2['Population'].loc[66]*np.exp((linear_func(prediction_range, *lambdaparams)*(prediction_range-2016))), color='purple', alpha=0.3)
axs.legend()
# plt.savefig('Computer Simulation/Numerical Methods/Assignment 2/non_constant_growth.pdf')
plt.show()
plt.close()

print('Name: David Lawton', '\n', 'Student Number: 22337087')