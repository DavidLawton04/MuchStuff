import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def f(x, t):
    return (1 + t)*x + 1 - 3*t + t**2

grid_points = np.meshgrid(np.linspace(0, 5, 10), np.linspace(-3, 3, 10))

fig, ax = plt.subplots()
fig.suptitle("Direction Field for $f(x,t) = \\frac{dx}{dt}$")

ax.set_xlabel("$t~(s)$")
ax.set_ylabel("$x~(m)$")
ax.quiver(grid_points[0], grid_points[1], 1, f(grid_points[1], grid_points[0]), scale=100)

plt.savefig("/home/dj-lawton/Documents/Junior Sophister/Computer Simulation/Numerical Methods/Assignment 3/Direction_Field.pdf")
plt.show()
plt.close()

# Parameters
t = 0
dt = 0.04 
x_0 = 0.0655
t_f_vals = [1.5, 2.5, 4]


#Simple Euler Method
def e_method(x_initial, t_final, dt):
    x = x_initial
    t = 0
    x_vals = [x]
    t_vals = [t]
    for i in range(int(t_final/dt)):
        x += f(x, t)*dt
        t += dt
        x_vals.append(x)
        t_vals.append(t)
    return x_vals, t_vals


# Plot the simple Euler method
fig, ax = plt.subplots(1,3)
fig.set_size_inches(12, 6)
fig.suptitle("Euler Method Approximation")

for i in range(len(t_f_vals)):
    x_vals, t_vals = e_method(x_0, t_f_vals[i], dt)
    grid_points = np.meshgrid(np.linspace(0, t_f_vals[i], int(len(t_vals)/5)), np.linspace(np.min(x_vals), np.max(x_vals), int(len(x_vals)/5)))
    ax[i].plot(t_vals, x_vals)
    ax[i].quiver(grid_points[0], grid_points[1], 1, f(grid_points[1], grid_points[0]), scale=50*np.max(x_vals))
    ax[i].set_title(f"$t_f = {t_f_vals[i]}s$")
    ax[i].set_xlabel("$t~(s)$")
    ax[i].set_ylabel("$x~(m)$")

plt.savefig("/home/dj-lawton/Documents/Junior Sophister/Computer Simulation/Numerical Methods/Assignment 3/Euler_Method_Approximation_0.04.pdf")
plt.show()
plt.close()


# Improved Euler method
def improved_euler_method(x_initial, t_final, dt):
    x = x_initial
    t = 0
    x_vals = [x]
    t_vals = [t]
    for i in range(int(t_final/dt)):
        x += (dt/2)*(f(x, t) + f(x + f(x, t)*dt, t + dt))
        t += dt
        x_vals.append(x)
        t_vals.append(t)
    return x_vals, t_vals


# 4th order Runge-Kutta method
def Runge_Kutta_4th_Order(x_initial, t_final, dt):
    x = x_initial
    t = 0
    x_vals = [x]
    t_vals = [t]
    for i in range(int(t_final/dt)):
        x1, t1 = x, t
        x2, t2 = x + (dt/2)*f(x1, t1), t + dt/2
        x3, t3 = x + (dt/2)*f(x2, t2), t + dt/2
        x4, t4 = x + dt*f(x3, t3), t + dt
        x += (dt/6)*(f(x1,t1) + 2*f(x2, t2) + 2*f(x3, t3) + f(x4, t4))
        t += dt
        x_vals.append(x)
        t_vals.append(t)
    return x_vals, t_vals


# Plotting the higher order approximations
fig, ax = plt.subplots(1,3)
fig.set_size_inches(18, 6)  
fig.suptitle("Higher Order Approximations")

for i in range(len(t_f_vals)):
    
    # Calling the methods
    e_method_x_vals, e_method_t_vals = e_method(x_0, t_f_vals[i], dt)
    improved_e_method_x_vals, improved_e_method_t_vals = improved_euler_method(x_0, t_f_vals[i], dt)
    runge_kutta_method_x_vals, runge_kutta_method_t_vals = Runge_Kutta_4th_Order(x_0, t_f_vals[i], dt)

    # Generating the direction field
    grid_points = np.meshgrid(np.linspace(0, t_f_vals[i], int(len(e_method_t_vals)/6)), np.linspace(np.min(runge_kutta_method_x_vals), np.max(e_method_x_vals), int(len(e_method_x_vals)/6)))


    ax[i].set_title(f"$t_f = {t_f_vals[i]}s$")

    ax[i].quiver(grid_points[0], grid_points[1], 1, f(grid_points[1], grid_points[0]), scale=40*np.max(e_method_x_vals))

    ax[i].plot(e_method_t_vals, e_method_x_vals, label='Euler Method')
    ax[i].plot(improved_e_method_t_vals, improved_e_method_x_vals, label='Improved Euler Method')
    ax[i].plot(runge_kutta_method_t_vals, runge_kutta_method_x_vals, label='Runge-Kutta 4th Order')

    ax[i].set_xlabel("$t~(s)$")
    ax[i].set_ylabel("$x~(m)$")

    ax[i].legend()

plt.savefig("/home/dj-lawton/Documents/Junior Sophister/Computer Simulation/Numerical Methods/Assignment 3/Higher_Order_Approximations_0.04.pdf")
plt.show()
plt.close()

print('Name: David Lawton', '\n', 'Student No.: 22337087')

