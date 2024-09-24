import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define Heron's Root Finding method.
def HeronRFM(x_n, a):
    x = 0.5 * (x_n + a / x_n)
    return x

# Define possible fitting function for convergence rate.
def fitfunc(x, a, b, c, d):
    return d*(x-a)**-3 - b*(x-a)**-2 +c

# Define constants: c = no. to find the square root of,
# iterations = no. of iterations to run.
c = 2
iterations = 15

# Define function to record iterations of Heron's Method.
def sqrroot(a, x0, j):
    x_n_vals = [[],[]]
    x = x0
    for i in range(j):
        x = HeronRFM(x, a)
        x_n_vals[0].append(x)
        x_n_vals[1].append(i+1)
    
    return x_n_vals

# Define function to analyse the results of Heron's Method using
# a range of initial guesses, and plots of the results.
def analysis(c_, iterations_, q_: int, d_: list):

    # Define lists to store results of Heron's Method.
    # 'vals' stores the results of Heron's Method for each initial guess.
    vals = []

    # 'calc_length' stores the no. of iterations required for each guess to converge.
    calc_length = [[],[]]
    for i in range(q_):
        vals.append(sqrroot(c_, d_[i], iterations_))
    for j in range(q_):
        for i in range(iterations):
            if vals[j][0][i] == vals[j][0][-1]:
                calc_length[0].append(d_[j]-np.sqrt(c_))
                calc_length[1].append(i)
                break
        print(calc_length)

    # Creating plot of results.
    fig, axs = plt.subplots(1,3)
    fig.set_figwidth(15)
    fig.set_figheight(5)
    fig.suptitle(f"Heron's Method, a = {c_}, Iterations = {iterations_}, range: sqrt({c_}) to {d_[-1]}")

    # Define colors for each initial guess, to allow better analysis of paths.
    # Better choices of color maps improve visibility of paths, convergence.
    colors = plt.cm.inferno_r(np.linspace(0, 1, q_))

    # First subplot: Iterations of x_n.
    axs[0].set_title(f"Iterations of x_n")
    for i in range(q_):
        axs[0].plot(vals[i][1], vals[i][0], color=colors[i])
    axs[0].set_xlabel("Iterations")
    axs[0].set_yscale("log")
    axs[0].set_ylabel("x_n")

    # Second subplot: Relative Error of x_n.
    axs[1].set_title(f"Relative Error of x_n")
    for i in range(q_):
        axs[1].plot(vals[i][1], (np.array(vals[i][0])**2)/c_, color=colors[i])
    axs[1].set_xlabel("Iterations")
    axs[1].set_yscale("log")
    axs[1].set_ylabel("Relative Error")
    
    # Idea of fitting function for convergence rate, unsuccesful, likely power law.

    # calc_length[0].insert(2, 0)
    # calc_length[1].insert(2, 0)
    # params, pcov = curve_fit(fitfunc, calc_length[0], calc_length[1], method = "lm", absolute_sigma=True, maxfev = 100000)
    # print("params:", params)
    # stdev =np.sqrt(np.diag(pcov))
    # print(stdev)

    # Third subplot: Convergence Rate.
    axs[2].set_title(f"Convergence Rate")
    #axs[2].plot(calc_length[0], fitfunc(np.array(calc_length[0]), *params), "r")
    axs[2].plot(calc_length[0], calc_length[1], "-b")
    axs[2].scatter(calc_length[0], calc_length[1], color="y")
    axs[2].set_xlabel("Initial Guess - sqrt(c)")
    axs[2].set_ylabel("Iterations to Convergence")


    plt.savefig(f"/home/dj-lawton/Documents/Junior Sophister/Computer Simulation/HeronRMF{c_}.pdf")
    plt.close()
    print(f"The square root of {c_} is {vals[1][0][-1]}.")

# Define range of initial guesses for Heron's Method.
inp = np.logspace(np.log10(np.sqrt(2)), 3, 50)

# Run analysis of Heron's Method for the range of initial guesses.
analysis(c,  iterations, len(inp), inp)