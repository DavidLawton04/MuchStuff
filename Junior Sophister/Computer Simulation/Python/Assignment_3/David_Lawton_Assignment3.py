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
                E -= spin * neighbors
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

        for _ in range(num_sweeps):
            self.sweep()
            magnetization_list.append(self.magnetization())
            magsquare_list.append(self.magnetization() ** 2)
            energy_list.append(self.energy())

        return np.mean(magnetization_list), np.mean(magsquare_list), np.mean(energy_list)



# Example usage:
model = IsingModel(size=10, temperature=2.0, h=1.0)
mean_mag, mean_magsquare, mean_energy = model.simulate(num_sweeps=1500)
magnetic_susceptibility = (mean_magsquare - mean_mag ** 2) * model.beta

# Added spins, sweeps, temperature print statements to show more results of the simulation
print("Number of spins:", model.size ** 2,"\n",
      "Number of sweeps:", model.size ** 2,"\n",
      "Temperature:", model.temperature,"\n",
      "Magnetization:", model.magnetization(),"\n",
      "Energy:", model.energy())

print("Mean Magnetization:", mean_mag,"\n",
      "Magnetic Susceptibility:", magnetic_susceptibility,"\n",
      "Mean Energy:", mean_energy)

# The AI did not test how many sweeps were required for convergence, so I added this
mag_sus_list = []
energy_list = []
mag_list = []

test_conv_list = np.logspace(1, 4, 9, endpoint=True)
for q in test_conv_list:
    mean_mag, mean_magsquare, mean_energy = model.simulate(num_sweeps=int(q))
    magnetic_susceptibility = (mean_magsquare - mean_mag ** 2) * model.beta
    energy_list.append(mean_energy)
    mag_sus_list.append(magnetic_susceptibility)
    mag_list.append(mean_mag)

fig, axs = plt.subplots(2)
fig.set_size_inches(5, 10)
fig.suptitle("Convergence of Metropolis Algorithm")

axs[0].set_title("$\\langle E \\rangle$ vs Number of Sweeps")
axs[0].scatter(test_conv_list, energy_list, color='yellow', alpha=0.9)
axs[0].plot(test_conv_list, energy_list, drawstyle='steps-mid')

axs[1].set_title("$\\langle M \\rangle$ vs Number of Sweeps")
axs[1].scatter(test_conv_list, mag_list, color='yellow', alpha=0.9)
axs[1].plot(test_conv_list, mag_list, drawstyle='steps-mid')

plt.show()

# Found convergence to a an acceptible level of accuracy at approximately 4000 sweeps
# It may be more efficient to use ~2000 sweeps, as the system on most runs has a reasonable level of convergence,
# however in using this more efficient choice, one gives up some accuracy.

# Next, plot mean of M and E, as well as magnetic susceptibility and heat capacity as a function of temperature


h_range = np.linspace(1, 5, 10)
temperature_vals = [1.0, 4.0]



for temperature in temperature_vals:
    mean_mags = []
    mean_energies = []
    mean_magsups = []
    mean_heatcaps = []
    fig, axs = plt.subplots(2,2)
    fig.suptitle(f"$\\langle M \\rangle$, $\\langle E \\rangle$, $\\chi$, and $C_v$ vs T, T={temperature}")

    for h in h_range:
        model = IsingModel(size=10, temperature=temperature, h=h)
        mean_mag, mean_magsquare, mean_energy = model.simulate(num_sweeps=4000)
        magnetic_susceptibility = (mean_magsquare - mean_mag ** 2) * model.beta
        heat_capacity = (mean_energy ** 2 - mean_energy ** 2) * model.beta ** 2

        mean_mags.append(mean_mag)
        mean_energies.append(mean_energy)
        mean_magsups.append(magnetic_susceptibility)
        mean_heatcaps.append(heat_capacity)
    
    axs[0,0].plot(h_range, mean_mags, label=f"T = {temperature}")
    axs[0,0].set_title("$\\langle M \\rangle$ vs $h$")
    axs[0,0].set_xlabel("$h$")
    axs[0,0].set_ylabel("$\\langle M \\rangle$")
    axs[0,0].legend()

    axs[0,1].plot(h_range, mean_energies, label=f"T = {temperature}")
    axs[0,1].set_title("$\\langle E \\rangle$ vs $h$")
    axs[0,1].set_xlabel("$h$")
    axs[0,1].set_ylabel("$\\langle E \\rangle$")
    axs[0,1].legend()

    axs[1,0].plot(h_range, mean_magsups, label=f"T = {temperature}")
    axs[1,0].set_title("$\\chi$ vs $h$")
    axs[1,0].set_xlabel("$h$")
    axs[1,0].set_ylabel("$\\chi$")
    axs[1,0].legend()

    axs[1,1].plot(h_range, mean_heatcaps, label=f"T = {temperature}")
    axs[1,1].set_title("$C_v$ vs $h$")
    axs[1,1].set_xlabel("$h$")
    axs[1,1].set_ylabel("$C_v$")
    axs[1,1].legend()

    plt.savefig(f'/home/dj-lawton/Documents/Junior Sophister/Computer Simulation/Python/Assignment_3/h_graph_T={temperature}.pdf')
    plt.show()

    

