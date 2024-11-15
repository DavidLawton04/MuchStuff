import numpy as np
import matplotlib.pyplot as plt


class IsingModel1D:

    # Initialises variables, and randomly assigns spins
    def __init__(self, num_spins, temperature, h=0.0):
        self.num_spins = num_spins
        # kB absorbed in temp param
        self.temperature = temperature
        self.spins = np.random.choice([-1, 1], size=num_spins)
        self.beta = 1.0 / temperature
        self.h = h

    # Defines total energy of system, J = 1
    def energy(self):
        return -np.sum(self.spins[:-1] * self.spins[1:])
    
    def energy_under_ext_field(self):
        return -np.sum(self.spins[:-1] * self.spins[1:]) - self.h * np.sum(self.spins)

    # Flips the spin at the given index
    def flip_spin(self, index):
        self.spins[index] *= -1

    # Defines the change in energy due to flipping a spin, 
    # as well as conditions for boundary cases
    def delta_energy(self, index):
        left_neighbor = self.spins[index - 1] if index > 0 else 0
        right_neighbor = self.spins[index + 1] if index < self.num_spins - 1 else 0
        return 2 * self.spins[index] * (left_neighbor + right_neighbor)

    # Metropolis step
    def metropolis_step(self):
        for _ in range(self.num_spins):
            index = np.random.randint(0, self.num_spins)
            delta_E = self.delta_energy(index)
            if delta_E < 0 or np.random.rand() < np.exp(-self.beta * delta_E):
                self.flip_spin(index)

    # Runs simulation for given no. of steps
    def simulate(self, num_steps):
        E_expec = 0
        M_expec = 0
        M_list = []
        E2_expec = 0
        M2_expec = 0

        for _, num in enumerate(np.linspace(0,num_steps ,num_steps)):
            self.metropolis_step()
            E_expec += self.energy_under_ext_field()
            M_expec += self.magnetization()
            M_list.append(self.magnetization()/num)
            E2_expec += self.energy_under_ext_field() ** 2
            M2_expec += self.magnetization() ** 2
        
        E_expec /= num_steps
        M_expec /= num_steps
        E2_expec /= num_steps
        M2_expec /= num_steps

        return [E_expec, M_expec, E2_expec, M2_expec], M_list

    # Total magnetisation of the system
    def magnetization(self):
        return np.sum(self.spins)


# Example usage
if __name__ == "__main__":
    num_spins = 100
    temperature = 2.0#*10**(8)
    num_steps = [400]
    # num_steps = np.logspace(100, 10000, 3, endpoint=True)
    h = temperature

    magnetisation = []
    for i, num_steps_val in enumerate(num_steps):

        model = IsingModel1D(num_spins, temperature, h=h)
        expectations, mag_averages = model.simulate(int(num_steps_val))

        # print(f'iteration {i} done:', a)
        # Note I add all but energy to print.
        # print(" Total No. of Steps:", model.num_spins, '\n', "Temperature", '\n', model.temperature, '\n', "Final magnetization:", model.magnetization(), '\n', "Final energy:", model.energy())

    fig, ax = plt.subplots()

    ax.plot(np.linspace(0,num_steps[0], num_steps[0]), mag_averages, drawstyle='steps-mid')
    ax.set_xlabel("Number of Steps")
    ax.set_ylabel("Total Magnetization")

    plt.show()
    plt.close()

    def find_convergence(mag_expec_list, conv_length):
        conv_matrix = np.zeros(conv_length)
        conv_matrix_prime = conv_matrix
        for i in range(len(mag_expec_list)):
            for j in range(conv_length):
                if np.isclose(mag_expec_list[i], mag_expec_list[i-j], atol=1e-3):
                    conv_matrix_prime[j] = 1
            if conv_matrix.all() == 1:
                print("Convergence reached at iteration:", i)
                convergence_iteration = i
                break
            else:
                conv_matrix_prime = conv_matrix
        return convergence_iteration
    
    iteration = find_convergence(mag_averages, 10)

    fig, ax = plt.subplots()

    h_vals
    ax.plot()


    