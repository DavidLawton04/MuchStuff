import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Below, each class corresponds to a different part of the assignment.
# Feel free to change the values of the class variables to test the code. 
# The code of each can be run by uncommenting the calling of the relevant modules in each instance.


class heronsRootMethod:

    # Define class variables.
    def __init__(self):
        self.c_ = 2
        self.iterations = 5

        # Define range of initial guesses for Heron's Method.
        # self.inp = np.logspace(np.log10(np.sqrt(2)), 4, 50)
        self.inp = (1, 2)
        self.length = len(self.inp)

    # Define Heron's Root Finding method.
    def HeronRFM(self, x_n, a):
        x = 0.5 * (x_n + a / x_n)
        return x

    # Define possible fitting function for convergence rate.
    def fitfunc(self):
        return self.d*(self.x-self.a)**-3 - self.b*(self.x-self.a)**-2 +self.c

    # Define function to record iterations of Heron's Method.
    def sqrroot(self):
        x = []
        y = []
        for j in range(self.length):
            x.append([self.inp[j]])
            y.append([0])
            for i in range(self.iterations):
                x[j].append(self.HeronRFM(x[j][-1], self.c_))
                y[j].append(i+1)
        return x, y


    # Define function to analyse the results of Heron's Method using
    # a range of initial guesses, and plots of the results.
    def analysis(self):

        vals, its = self.sqrroot()
        print(vals, its)

        # Creating plot of results.
        fig, axs = plt.subplots(1,3)
        fig.set_figwidth(17)
        fig.set_figheight(5)
        fig.suptitle(f"Heron's Method, $a$ = ${self.c_}$, Iterations $= {self.iterations}$, Starting guesses: ${self.inp}$")

        # Define colors for each initial guess, to allow better analysis of paths.
        # Better choices of color maps improve visibility of paths, convergence
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, self.length))
        # First subplot: Iterations of x_n.
        axs[0].set_title(f"Iterations of $x_n$")
        for i in range(self.length):
            axs[0].plot(its[i], vals[i], color=colors[i], marker='o', markersize=5, linestyle='-', linewidth=1, drawstyle='steps-mid')
        axs[0].set_xlabel("Iterations")
        axs[0].axhline(y=np.sqrt(self.c_), color="red", linestyle="--")
        # axs[0].text(0, self.c_ - 0.1, f"{self.c_}", color="red", ha="right", va="center")
        axs[0].set_ylabel("$x_n$")

        # Second subplot: Relative Error of x_n.
        axs[1].set_title(f"Relative Error of $x_n$")
        for i in range(self.length):
            axs[1].plot(its[i], (np.abs(self.c_ - (np.array(vals[i])**2)))/self.c_, color=colors[i], marker='o', markersize=5, linestyle='-', linewidth=1, drawstyle='steps-mid')
        axs[1].set_xlabel("Iterations")
        axs[1].set_yscale("log")
        axs[1].set_ylabel("Relative Error")

        axs[2].set_title("Convergence of square to input value")
        for i in range(self.length):
            axs[2].plot(its[i], (np.array(vals[i])**2), color=colors[i], marker='o', markersize=5, linestyle='-', linewidth=1, drawstyle='steps-mid')
        axs[2].set_xlabel("Iterations")
        axs[2].set_ylabel("$(x_n)^2$")
        axs[2].axhline(y=self.c_, color="red", linestyle="--")
        axs[2].text(0, self.c_ - 0.1, f"{self.c_}", color="red", ha="right", va="center")
        

        plt.savefig(f"/home/dj-lawton/Documents/Junior Sophister/Computer Simulation/HeronRMF{self.c_}.pdf")
        plt.close()
        print(f"The square root of {self.c_} is {vals[0][-1]}.")
    

class UnderOverFlow:

    def __init__(self):
        self.under = 1.0
        self.over = 1.0

    # Define function to calculate underflow.
    def underflow(self):
        i = []
        u = []
        while self.under > 0:
            self.under = self.under/2
            u.append(self.under)
            i.append(len(u))
        print(f"The underflow occurred after {i[-1]} iterations.")
        print(f"The underflow values are {u[-2]}.")
        # plt.plot(i[:-2],u[:-2])
        # plt.yscale("log")
        # plt.xlim(1, 10e-340)
        # plt.savefig("/home/dj-lawton/Documents/Junior Sophister/Computer Simulation/Underflow.pdf")
    # x = underflow()

    # Define function to calculate overflow.
    def overflow(self):
        i = []
        o = []
        for j in range(3000):
            self.over = self.over*2
            o.append(self.over)
            i.append(j)
            if np.isinf(self.over) == True:
                self.over = 0
                break
        print(f"The overflow occurred after {i[-1]} iterations.")
        print(f"The overflow value is {o[-2]}.")
        # plt.plot(i[:-2],o[:-2])
        # plt.yscale("log")
        # plt.savefig("/home/dj-lawton/Documents/Junior Sophister/Computer Simulation/Overflow.pdf")

class Precision:

    def __init__(self):
        self.precision = 1.0
        self.compprecision = 1.0j

    # Define function to calculate precision.
    def precision(self):
        i = 0
        while self.precision + 1 != 1:
            self.precision = self.precision/2
            i += 1
        print(f"The real precision is {self.precision}.")
        print(f"The complex precision estimate occurred after {i} iterations.")
        # plt.plot(i, precision)
        # plt.yscale("log")
        # plt.savefig("/home/dj-lawton/Documents/Junior Sophister/Computer Simulation/Precision.pdf")
    # x = precision()

    def complexprecision(self):
        j = 0
        while self.compprecision + 1j != 1j:
            self.compprecision = self.compprecision/2
            j += 1
        print(f"The complex precision is {self.compprecision}.")
        print(f"The complex precision estimate occurred after {j} iterations.")

class difference_methods:

    def __init__(self):
        self.times = np.array([0.1, 1, 100])
        self.h = 0.01
        self.iterations = 35
        self.initial_h = self.h

    def difference_methods_cos(self):
        fwd = (np.cos(self.times + self.h) - np.cos(self.times))/self.h
        central = (np.cos(self.times + self.h) - np.cos(self.times - self.h))/(2*self.h)

        return fwd, central
    
    def difference_method_exp(self):
        fwd = (np.exp(self.times + self.h) - np.exp(self.times))/self.h
        central = (np.exp(self.times + self.h) - np.exp(self.times - self.h))/(2*self.h)
        return fwd, central,
    
    def analysis_cos(self):

        length = len(self.times)
        hvals = []
        fwdiff = np.empty((length, self.iterations), float)
        ctdiff = np.empty((length, self.iterations), float)
        colordict = {"central": "blue", "fwd": "red"}
        self.h = self.initial_h
        for j in range(self.iterations):
            
            self.h = self.h/2
            
            fwd, central = self.difference_methods_cos()
            hvals.append(self.h)
            # print(len(hvals))

            for i in range(length):
                fwdiff[i, j] = fwd[i]
                ctdiff[i, j] = central[i]
                # print(fwdiff)
                # print(bwdiff)
            # print(fwdiff.shape)
        
        fig, axs = plt.subplots(2, 3)
        fig.set_figwidth(18)
        fig.set_figheight(10)
        fig.suptitle("Forward and Central Difference Methods for $f(x) = \\cos(x)$")
        
        for i in range(length):
            axs[0, i].set_title(f"$t = {self.times[i]}$")
            axs[0, i].plot(hvals, ctdiff[i, 0:], label="Central Difference")
            axs[0, i].plot(hvals, fwdiff[i, 0:], label="Forward Difference")
            axs[0, i].set_xlabel("$h$")
            axs[0, i].set_xscale("log")
            axs[0, i].set_ylabel(f"Derivative at ${self.times[i]}$ of $\\cos(x)$")
            axs[0, i].legend()

        for i in range(length):
            axs[1, i].set_title(f"Error at $t = {self.times[i]}$")
            axs[1, i].plot(hvals, np.abs(ctdiff[i, 0:] + np.sin(self.times[i])), marker="o", markersize=5, color=colordict["central"], label="Central Difference")
            axs[1, i].plot(hvals, np.abs(fwdiff[i, 0:] + np.sin(self.times[i])), marker="o", markersize=5, color=colordict["fwd"], label="Forward Difference")
            axs[1, i].set_xlabel("$h$")
            axs[1, i].set_xscale("log")
            axs[1, i].set_ylabel(f"Error at ${self.times[i]}$ of $\\cos(x)$")
            axs[1, i].set_yscale("log")
            axs[1, i].legend()

        plt.savefig("/home/dj-lawton/Documents/Junior Sophister/Computer Simulation/DifferenceMethodsCos.pdf")

        for j in range(length):
            print(f"For the forward difference method, with step size {hvals[5]}, the derivative of $\\cos(x)$ at $t = {self.times[j]}$ is ${fwdiff[j, 5]}$,\n and the error is {np.abs(fwdiff[j, 5] - np.sin(self.times[j]))}.")
            print(f"For the Central difference method, with step size {hvals[5]}, the derivative of $\\cos(x)$ at $t = {self.times[j]}$ is ${ctdiff[j, 5]}$, \n and the error is {np.abs(ctdiff[j, 5] - np.sin(self.times[j]))}.")
    
    def analysis_exp(self):

        length = len(self.times)
        hvals = []
        fwdiff = np.empty((length, self.iterations), float)
        ctdiff = np.empty((length, self.iterations), float)
        colordict = {"central": "blue", "fwd": "red"}
        self.h = self.initial_h
        for j in range(self.iterations):
            
            self.h = self.h/2
            
            fwd, central = self.difference_method_exp()
            hvals.append(self.h)
            # print(len(hvals))

            for i in range(length):
                fwdiff[i, j] = fwd[i]
                ctdiff[i, j] = central[i]
                # print(fwdiff)
                # print(bwdiff)
            # print(fwdiff.shape)
        
        fig, axs = plt.subplots(2, length)
        fig.set_figwidth(18)
        fig.set_figheight(10)
        fig.suptitle("Forward and Central Difference Methods for $f(x) = e^x$")
        
        for i in range(length):
            axs[0, i].set_title(f"$t = {self.times[i]}$")
            axs[0, i].plot(hvals, ctdiff[i, 0:], label="Central Difference")
            axs[0, i].plot(hvals, fwdiff[i, 0:], label="Forward Difference")
            axs[0, i].set_xlabel("$h$")
            axs[0, i].set_xscale("log")
            axs[0, i].set_ylabel(f"Derivative at ${self.times[i]}$ of $e^x$")
            axs[0, i].legend()

        for i in range(length):
            axs[1, i].set_title(f"Error at $t = {self.times[i]}$")
            axs[1, i].plot(hvals, np.abs(ctdiff[i, 0:] - np.exp(self.times[i])), marker="o", markersize=5, color=colordict["central"], label="Central Difference")
            axs[1, i].plot(hvals, np.abs(fwdiff[i, 0:] - np.exp(self.times[i])), marker="o", markersize=5, color=colordict["fwd"], label="Forward Difference")
            axs[1, i].set_xlabel("$h$")
            axs[1, i].set_xscale("log")
            axs[1, i].set_yscale("log")

        plt.savefig("/home/dj-lawton/Documents/Junior Sophister/Computer Simulation/DifferenceMethodsExp.pdf")
        

    



# Run analysis of Heron's Method for the range of initial guesses.
# Create an instance of the heronsRootMethod class
heron = heronsRootMethod()

# Call the analysis method on the instance
# heron.analysis()

underflow = UnderOverFlow()
# underflow.underflow()
# underflow.overflow()

precision = Precision()
# precision.precision()
# precision.complexprecision()

diff = difference_methods()
diff.analysis_cos()
diff.analysis_exp()



print("Student Number: 22337087\nName: David Lawton")