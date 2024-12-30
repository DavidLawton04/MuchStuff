# include <cmath>
# include <random>
# include <vector>
# include <iostream>
# include <fstream>

using namespace std;

// Ising Model
vector<vector<int>> lattice_init(int N) {
    vector<vector<int>> lattice(N, vector<int>(N, 1));
    for (int i; i<N; i++) {
        for (int j; j<N; j++) {
            lattice[i][j] = lattice[i][j] * (2*rand()%2 - 1);
        }
    }
    return lattice;
}

/* E = J \times \sum_{i,j=1}^{N}( s_{ij} * ( s_{i(j+1 % N)} + s_{(i+1 % N)j}) ) 
        + h \times \sum_{i,j=0}^{N}( s_{ij} )*/
float delta_E(vector<vector<int>>& lattice, float h, int i, int j) {
    int N = lattice.size();
    int neighbours = lattice[i][(j+1)%N] + lattice[(i+1)%N][j] + lattice[(i-1)%N][j] + lattice[i][(j-1)%N];

    float delta_E = - 2 * lattice[i][j] * (neighbours + h);

    return delta_E;
}

float total_E(vector<vector<int>>& lattice, float h, int N) {
    float total_E = 0;
    for (int i; i<N; i++) {
        for (int j; j<N; j++) {
            total_E += lattice[i][j] * (lattice[i][(j+1)%N] + lattice[(i+1)%N][j] + h);
        }
    }
    return total_E;
}

float total_mag(vector<vector<int>>& lattice, float h, float N) {
    float total_magnetisation = 0;

    for (int i;i<N;i++) {
        for (int j; j<N; j++) {
            total_magnetisation += lattice[i][j];
        }
    }

    total_magnetisation = total_magnetisation/N;
    return total_magnetisation;
}


// Temperature is k_B * T
// Function to perform one metropolis step at a given index.
void metropolis_step(vector<vector<int>>& lattice, vector<int> indices, float T, float h) {
    int i = indices[0];
    int j = indices[1];
    float dE = delta_E(lattice, h, i, j);

    if ((dE<0) || (rand()/RAND_MAX<exp(-dE/T))) {
        lattice[i][j] *= -1;
    }
}

// Function to sweep over lattice; first func randomly, second sequentially.
vector<float> rand_sweep(vector<vector<int>>& lattice, int N, float h, float T) {
    vector<float> output = {};

    for (int n = 0;n<pow(N,2);n++) {
        vector<int> indices = {rand()%N, rand()%N};
        metropolis_step(lattice, indices, T, h);
    }

    output.push_back(total_E(lattice, h, N));
    output.push_back(total_mag(lattice, h, N));

    return output;
}

vector<float> seq_sweep(vector<vector<int>>& lattice, int N, float h, float T) {
    vector<float> output = {};

    for (int i = 0;i<N;i++) {
        for (int j;j<N;j++) {
            vector<int> indices = {i,j};
            metropolis_step(lattice, indices, T, h);
        }
    }

    output.push_back(total_E(lattice, h, N));
    output.push_back(total_mag(lattice, h, N));

    return output;
}

float mean_of_vector(vector<float> vec) {
    float mean = 0;
    int N = vec.size();
    for (int i = 0;i<N;i++){
        mean += vec[i];
    }
    mean /= N;
    return mean;
}

void simulation(float h, float T, int N, int max_num_sweeps, vector<float> (*sweep_func)(vector<vector<int>>&, int, float, float)) {
    vector<vector<int>> lattice = lattice_init(N);
    vector<float> energies = {};
    vector<float> mags = {};

    // code to test number of sweeps required
    ofstream outfile ("/home/dj-lawton/Documents/Junior Sophister/Learning_C++/convergence.csv");
    outfile << "mean energies," << "mean magnetisations" << endl;
    for (int i = 0;i<max_num_sweeps;i++) {
        vector<float> sweep_out = sweep_func(lattice, N,h,T);
        energies.push_back(sweep_out[0]);
        mags.push_back(sweep_out[1]);
        outfile << mean_of_vector(energies) <<","<< mean_of_vector(mags) << endl;
    }
    outfile.close();
}

int main() {
    simulation(1, 1, 10, 15000, &seq_sweep);
}