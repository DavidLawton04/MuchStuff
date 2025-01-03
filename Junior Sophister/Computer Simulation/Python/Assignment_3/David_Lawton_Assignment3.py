import numpy as np
import random
import matplotlib.pyplot as plt

class IsingModel:

    # Copilot created the initilisation method to define the size, temperature and initial lattice as attributes of the class.
    # I added the h parameter to take into account an external magnetic field perpendicular to the 2D model (positive or negative).
    def __init__(self, size, temperature, h=0):
        self.size = size
        self.temperature = temperature
        self.lattice = np.random.choice([-1, 1], size=(size, size))
        self.beta = 1.0 / temperature
        self.h = h

    # The AI did not define a spin flip method, but instead just multiplied the spin by -1 to flip it.

    # It created this method to find the change in energy due to the flipping of one spin
    # Although it did not include the external magnetic field in this calculation, and I have added it in.
    def _delta_energy(self, i, j):
        """Calculate the change in energy if the spin at (i, j) is flipped."""
        spin = self.lattice[i, j]
        neighbors = self.lattice[(i+1) % self.size, j] + self.lattice[i, (j+1) % self.size] + \
                    self.lattice[(i-1) % self.size, j] + self.lattice[i, (j-1) % self.size]
        return 2 * spin * neighbors + 2 * self.h * spin

    # The total magnetisation and energy of the lattice
    def magnetization(self):
        """Calculate the magnetization of the lattice."""
        return np.sum(self.lattice)

    def energy(self):
        """Calculate the energy of the lattice."""
        E = 0
        for i in range(self.size):
            for j in range(self.size):
                spin = self.lattice[i, j]
                neighbors = self.lattice[(i+1) % self.size, j] + self.lattice[i, (j+1) % self.size]
                E -= spin * neighbors + self.h * spin
        return E

    # The Metropolis algorithm here is used to minimise the energy, as it accepts any change lowering energy, and accepts
    # those increasing energy randomly with proability of acceptance depending on a function that is defined e^(-delta_E / (k_B * T)),
    # where k_B is Boltzmann's constant.

    # Where the AI originally chose to flip probabilistically L^2 random spins, I altered this code to flip 
    # probabalistically all spins, sweeping over the lattice.
    
    # AI's original code for each sweep
    def step(self):
        """Perform a single Metropolis step."""
        for _ in range(self.size ** 2):
            i = random.randint(0, self.size - 1)
            j = random.randint(0, self.size - 1)
            delta_E = self._delta_energy(i, j)
            if delta_E < 0 or random.random() < np.exp(-delta_E * self.beta):
                self.lattice[i, j] *= -1

    # My altered code, taking inspiration from the AI's original code, but sweeping over lattice such that all are flipped
    def sweep(self):
        for i, j in np.ndindex(self.lattice.shape):
            delta_E = self._delta_energy(i, j)
            if delta_E < 0 or random.random() < np.exp(-delta_E * self.beta):
                self.lattice[i, j] *= -1

    # Next the model is simulated for a number of sweeps, to be specified later, and store average magnetization and energy.
    # The AI originally ran many single random steps and stored the total magnetisation and energy after the final step.
    def simulate(self, num_sweeps):
        """Run the simulation for a given number of steps."""
        magnetization_list = []
        magsquare_list = []
        energy_list = []
        ensquare_list = []
        for _ in range(num_sweeps):
            self.sweep()
            magnetization_list.append(self.magnetization())
            magsquare_list.append(self.magnetization() ** 2)
            energy_list.append(self.energy())
            ensquare_list.append(self.energy() ** 2)

        exp_mag = np.mean(magnetization_list)
        exp_magsquare = np.mean(magsquare_list)
        exp_energy = np.mean(energy_list)
        exp_ensquare = np.mean(ensquare_list)

        magnetic_susceptibility = (exp_magsquare - exp_mag ** 2) * self.beta
        heat_capacity = (exp_ensquare - exp_energy ** 2) * self.beta ** 2 / self.size ** 2
        return exp_mag, exp_energy, magnetic_susceptibility, heat_capacity



# Example usage:
model = IsingModel(size=10, temperature=2.0, h=1.0)
exp_magnetization, exp_energy, magnetic_susceptibility, heat_capacity = model.simulate(num_sweeps=1500)

# Added spins, sweeps, temperature print statements to show more results of the simulation
print("Number of spins:", model.size ** 2,"\n",
      "Number of sweeps:", model.size ** 2,"\n",
      "Temperature:", model.temperature, " J","\n",
      "Magnetization:", model.magnetization(),"\n",
      "Energy:", model.energy(), " J")

print("Mean Magnetization:", exp_magnetization,"\n",
      "Magnetic Susceptibility:", magnetic_susceptibility, " J","\n",
      "Mean Energy:", exp_energy, "J","\n",
      "Heat Capacity:", heat_capacity)

# The AI did not test how many sweeps were required for convergence, so I added this
mag_sus_list = []
energy_list = []
mag_list = []

test_conv_array = np.concatenate((np.linspace(100,1000,9),np.linspace(1000, 10000, 10, endpoint=True)))

for q in test_conv_array:
    mean_mag, mean_energy, magnetic_susceptibility, heat_capacity = model.simulate(num_sweeps=int(q))
    energy_list.append(mean_energy)
    mag_list.append(mean_mag)

fig, axs = plt.subplots(1,2)
fig.set_size_inches(10,7)
fig.suptitle("Convergence of Metropolis Algorithm")

axs[0].set_title("$\\langle E \\rangle$ vs Number of Sweeps")
axs[0].set_ylabel("$\\langle E \\rangle$ (J)")
axs[0].scatter(test_conv_array, energy_list, color='orange', alpha=0.9)
axs[0].plot(test_conv_array, energy_list, drawstyle='steps-mid')

axs[1].set_title("$\\langle M \\rangle$ vs Number of Sweeps")
axs[1].set_ylabel("$\\langle M \\rangle$")
axs[1].scatter(test_conv_array, mag_list, color='orange', alpha=0.9)
axs[1].plot(test_conv_array, mag_list, drawstyle='steps-mid')

plt.savefig('/home/dj-lawton/Documents/Junior Sophister/Computer Simulation/Python/Assignment_3/convergence.pdf')
plt.show()

# Found convergence to a an acceptible level of accuracy at approximately 4000 sweeps.
# It may be more efficient to use ~2000 sweeps, as the system on most runs has a reasonable level of convergence,
# however in using this more efficient choice, one gives up some accuracy.

# Quickly define a function to plot the graphs, to clean up code.
def graphing_function(ax, x, y, title, xlabel, ylabel, constant):
    ax.plot(x, y, marker='o', label=constant)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

# # Next, plot mean of M and E, as well as magnetic susceptibility and heat capacity as a function of temperature

h_range = np.linspace(0.1, 4, 10)
temperature_vals = [1.0, 4.0]

for temperature in temperature_vals:
    mean_mags = []
    mean_energies = []
    mean_magsups = []
    mean_heatcaps = []
    fig, axs = plt.subplots(2,2)
    fig.set_size_inches(10, 12)
    fig.suptitle(f"$\\langle M \\rangle$, $\\langle E \\rangle$, $\\chi$, and $C_v$ vs $h$, T={temperature}")

    for h in h_range:
        model = IsingModel(size=10, temperature=temperature, h=h)
        exp_mag, exp_energy, mag_sus, heat_cap = model.simulate(num_sweeps=2000)

        mean_mags.append(exp_mag)
        mean_energies.append(exp_energy)
        mean_magsups.append(mag_sus)
        mean_heatcaps.append(heat_cap)
    
    graphing_function(axs[0,0], h_range, mean_mags, "$\\langle M \\rangle$ vs $h$", "$h$ (J)", "$\\langle M\\rangle$", f"T = {temperature}")
    graphing_function(axs[0,1], h_range, mean_energies, "$\\langle E \\rangle$ vs $h$", "$h$ (J)", "$\\langle E\\rangle$ (J)", f"T = {temperature}")
    graphing_function(axs[1,0], h_range, mean_magsups, "$\\chi$ vs $h$", "$h$ (J)", "$\\chi$ ($J^{-1}$)", f"T = {temperature}")
    graphing_function(axs[1,1], h_range, mean_heatcaps, "$C_v$ vs $h$", "$h$ (J)", "$C_v$", f"T = {temperature}")

    plt.savefig(f'/home/dj-lawton/Documents/Junior Sophister/Computer Simulation/Python/Assignment_3/h_graph_T={temperature}.pdf')
    plt.show()
    plt.close()

# We now simulate the system for varying temperatures, with external magnetic field set to h=0
temperature_vals = np.linspace(1.0, 4.0, 10)
h = 0

mag_list = []
energy_list = []
mag_sus_list = []
heat_cap_list = []

fig, axs = plt.subplots(2,2)
fig.set_size_inches(10, 12)
fig.suptitle("$\\langle M \\rangle$, $\\langle E \\rangle$, $\\chi$, and $C_v$ vs $T$, $h=0$")

for temp in temperature_vals:
    model = IsingModel(size=10, temperature=temp, h=h)
    exp_mag, exp_energy, mag_sus, heat_cap = model.simulate(num_sweeps=2000)
    mag_list.append(exp_mag)
    energy_list.append(exp_energy)
    mag_sus_list.append(mag_sus)    
    heat_cap_list.append(heat_cap)

graphing_function(axs[0,0], temperature_vals, mag_list, "$\\langle M \\rangle$ vs $T$", "$T$ (J)", "$\\langle M\\rangle$", f"h = {h}")
graphing_function(axs[0,1], temperature_vals, energy_list, "$\\langle E \\rangle$ vs $T$", "$T$ (J)", "$\\langle E\\rangle$ (J)", f"h = {h}")
graphing_function(axs[1,0], temperature_vals, mag_sus_list, "$\\chi$ vs $T$", "$T$ (J)", "$\\chi$ ($J^{-1}$)", f"h = {h}")
graphing_function(axs[1,1], temperature_vals, heat_cap_list, "$C_v$ vs $T$", "$T$ (J)", "$C_v$", f"h = {h}")

plt.savefig(f'/home/dj-lawton/Documents/Junior Sophister/Computer Simulation/Python/Assignment_3/T_graph_h={h}.pdf')
plt.show()
plt.close()

# Critical Temperature: M goes to 0, inflection point of E, peak of C_v, chi.
# Observed to be approx. 2.3. for h=0.

# AI prompt:
# Now plot chi, C_v as function of system size, for T=2.3, h=0

# Define a range of system sizes
# AI originally, 5, 10-50, I changed to include more poijts between 10,30
system_sizes = np.array([10,12,15,17,20,22,25,27,30])
temperature = 2.4
h = 0

chi_values = []
cv_values = []

for size in system_sizes:
    model = IsingModel(size=size, temperature=temperature, h=h)
    exp_mag, exp_energy, mag_sus, heat_cap = model.simulate(num_sweeps=2000)
    chi_values.append(mag_sus)
    cv_values.append(heat_cap)

# Plot chi and Cv as a function of system size
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle(f"$\\chi$ and $C_v$ vs System Size, T={temperature}, h={h}")

axs[0].plot(system_sizes**2, chi_values, marker='o', linestyle='-', color='b')
axs[0].set_title("$\\chi$ vs System Size")
axs[0].set_xlabel("System Size")
axs[0].set_ylabel("$\\chi$ ($J^{-1}$)")

axs[1].plot(system_sizes**2, cv_values, marker='o', linestyle='-', color='r')
axs[1].set_title("$C_v$ vs System Size")
axs[1].set_xlabel("System Size")
axs[1].set_ylabel("$C_v$")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig(f'/home/dj-lawton/Documents/Junior Sophister/Computer Simulation/Python/Assignment_3/chi_cv_size.pdf')
plt.show()
