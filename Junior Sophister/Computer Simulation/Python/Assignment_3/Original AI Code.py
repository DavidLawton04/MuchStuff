import numpy as np
import random

class IsingModel:
    def __init__(self, size, temperature):
        self.size = size
        self.temperature = temperature
        self.lattice = np.random.choice([-1, 1], size=(size, size))
        self.beta = 1.0 / temperature

    def _delta_energy(self, i, j):
        """Calculate the change in energy if the spin at (i, j) is flipped."""
        spin = self.lattice[i, j]
        neighbors = self.lattice[(i+1) % self.size, j] + self.lattice[i, (j+1) % self.size] + \
                    self.lattice[(i-1) % self.size, j] + self.lattice[i, (j-1) % self.size]
        return 2 * spin * neighbors

    def step(self):
        """Perform a single Metropolis step."""
        for _ in range(self.size ** 2):
            i = random.randint(0, self.size - 1)
            j = random.randint(0, self.size - 1)
            delta_E = self._delta_energy(i, j)
            if delta_E < 0 or random.random() < np.exp(-delta_E * self.beta):
                self.lattice[i, j] *= -1

    def simulate(self, steps):
        """Run the simulation for a given number of steps."""
        for _ in range(steps):
            self.step()

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

# Example usage:
# model = IsingModel(size=10, temperature=2.0)
# model.simulate(steps=1000)
# print("Magnetization:", model.magnetization())
# print("Energy:", model.energy())