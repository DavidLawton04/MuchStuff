# include <cmath>
# include <random>
# include <vector>
# include <iostream>
# include <fstream>
# include <numeric>
# include <algorithm>
# include <list>
# include <sstream>
# include <cstdlib>
# include <fstream>

using namespace std;

class IsingModel {
public:
    IsingModel(int N, float T, float h);
    double delta_E(int i, int j);
    double total_E();
    double total_mag();
    double magnetic_susceptibility();
    double specific_heat();
    void metropolis_step(int i, int j);
    void seq_sim(int num_sweeps);
    void rand_sim(int num_sweeps);
    int conv_seq_sim(int num_max_sweeps, int tol);
    int conv_rand_sim(int max_num_sweeps, int tol);
    void lattice_init();
    double EnergyExpValue();
    double MagExpValue();

private:
    float h;
    int N;
    float T;
    float B;
    vector<vector<int>> lattice;
    double mean_of_vector(const vector<double>& vec);
    vector<double> vec_squared(const vector<double>& vec);
    vector<double> seq_sweep();
    vector<double> rand_sweep();
    vector<double> energies;
    vector<double> mags;

};
IsingModel::IsingModel(int N, float T, float h) {
    this->h = h;
    this->T = T;
    this->N = N;
    this->B = 1 / T; 
    lattice_init();

    // this->lattice = lattice_init();
}

void IsingModel::lattice_init() {
    vector<vector<int>> lattice(N, vector<int>(N, 1));
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            lattice[i][j] *= (2 * (rand() % 2) - 1);
        }
    }
    this->lattice = lattice;
}

/* E = J \times \sum_{i,j=1}^{N}( s_{ij} * ( s_{i(j+1 % N)} + s_{(i+1 % N)j}) ) 
        + h \times \sum_{i,j=0}^{N}( s_{ij} )*/
double IsingModel::delta_E(int i, int j) {
    int neighbours = lattice[i][(j+1)%N] + lattice[(i+1)%N][j] + lattice[(i-1+N)%N][j] + lattice[i][(j-1+N)%N];
    double delta_E = 2 * lattice[i][j] * (neighbours + h);

    return delta_E;
}

double IsingModel::total_E() {
    double total_E = 0;
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            total_E += -lattice[i][j] * (lattice[i][(j+1)%N] + lattice[(i+1)%N][j] + h);
        }
    }
    return total_E;
}

double IsingModel::total_mag() {
    double total_magnetisation = 0;
    for (int i=0;i<N;i++) {
        for (int j=0; j<N; j++) {
            total_magnetisation += lattice[i][j];
        }
    }
    total_magnetisation = total_magnetisation/(N*N);
    return total_magnetisation;
}

// Temperature is k_B * T
// Function to perform one metropolis step at a given index.
void IsingModel::metropolis_step(int i, int j) {
    float dE = delta_E(i, j);
    if ((dE<0) || ((static_cast<double>(rand())/RAND_MAX)<(exp(-dE/T)))) {
        lattice[i][j] *= (-1);
    }
}

// Function to sweep over lattice; first func randomly, second sequentially.
vector<double> IsingModel::rand_sweep() {
    vector<double> output = {};
    for (int n = 0;n<pow(N,2);n++) {
        int i = rand()%N;
        int j = rand()%N;
        metropolis_step(i,j);
    }
    output.push_back(total_E());
    output.push_back(total_mag());
    return output;
}

vector<double> IsingModel::seq_sweep() {
    vector<double> output = {};
    for (int i = 0;i<N;i++) {
        for (int j=0;j<N;j++) {
            metropolis_step(i,j);
        }
    }
    output.push_back(total_E());
    output.push_back(total_mag());
    return output;
}

double IsingModel::mean_of_vector(const vector<double>& vec) {
    double mean = accumulate(vec.begin(), vec.end(), 0.0f)/vec.size();
    return mean;
}

vector<double> IsingModel::vec_squared(const vector<double>& vec) {
    vector<double> vec_squared(vec.size());
    transform(vec.begin(), vec.end(), vec_squared.begin(), [](double value) {return value * value; });
    return vec_squared;
}

double IsingModel::EnergyExpValue() {
    double mean_energy = mean_of_vector(energies);
    return mean_energy;
}

double IsingModel::MagExpValue() {
    double mean_mag = mean_of_vector(mags);
    return mean_mag;
}

double IsingModel::magnetic_susceptibility() {
    double mag_sus = (mean_of_vector(vec_squared(mags))-pow(MagExpValue(),2))*B;
    return mag_sus;
}

double IsingModel::specific_heat() {
    double C_v = (mean_of_vector(vec_squared(energies))-pow(EnergyExpValue(),2))*pow(B,2)/pow(N,2);
    return C_v;
}

int IsingModel::conv_seq_sim(int max_num_sweeps, int tol) {
    vector<double> energies = {};
    vector<double> mags = {};
    vector<double> nrgs = {};
    int convergence=0;

    // code to test number of sweeps required
    ofstream outfile ("/home/dj-lawton/Documents/Junior Sophister/Learning_C++/sequential_sweep_convergence.csv");
    outfile <<"index,"<< "mean energies," << "mean magnetisations" << endl;
    for (int i = 0;i<max_num_sweeps;i++) {
        vector<double> sweep_out = seq_sweep();
        energies.push_back(sweep_out[0]);
        mags.push_back(sweep_out[1]);
        nrgs.push_back(mean_of_vector(energies));
        outfile << i+1 << "," << nrgs[i] <<","<< mean_of_vector(mags) << endl;
        if ((i!=0)&&(i!=1)&&(abs(-200 - nrgs[i])<pow(10,-tol))) {
            cout << "Convergence to given tolerance after "<< i << " iterations"<<endl;
            convergence = i;
            break;
        }
    }
    outfile.close();
    this->energies = energies;
    this->mags = mags;
    return convergence;
}

int IsingModel::conv_rand_sim(int max_num_sweeps, int tol) {
    vector<double> energies = {};
    vector<double> mags = {};
    vector<double> nrgs = {};
    int convergence=0;
    // code to test number of sweeps required
    ofstream outfile ("/home/dj-lawton/Documents/Junior Sophister/Learning_C++/random_sweep_convergence.csv");
    outfile <<"index,"<< "mean energies," << "mean magnetisations" << endl;
    for (int i = 0;i<max_num_sweeps;i++) {
        vector<double> sweep_out = seq_sweep();
        energies.push_back(sweep_out[0]);
        mags.push_back(sweep_out[1]);
        nrgs.push_back(mean_of_vector(energies));
        outfile << i+1 << "," << nrgs[i] <<","<< mean_of_vector(mags) << endl;
        if ((i!=0)&&(abs(-200-nrgs[i])<pow(10,-tol))) {
            cout << "Convergence to given tolerance after "<< i << " iterations"<<endl;
            convergence = i;
            break;
        }
    }
    outfile.close();
    this->energies = energies;
    this->mags = mags;
    return convergence;
}

void IsingModel::seq_sim(int max_num_sweeps) {
    vector<double> energies;
    vector<double> mags;
    for (int i=0;i<max_num_sweeps;i++) {
        vector<double> out = seq_sweep();
        energies.push_back(out[0]);
        mags.push_back(out[1]);
    }
    this->energies = energies;
    this->mags = mags;
}

void IsingModel::rand_sim(int max_num_sweeps) {
    vector<double> energies;
    vector<double> mags;
    for (int i=0;i<max_num_sweeps;i++) {
        vector<double> out = rand_sweep();
        energies.push_back(out[0]);
        mags.push_back(out[1]);
    }
    this->energies = energies;
    this->mags = mags;
}

void Const_T_Varying_h(float T_value, int num_data_points, float h_max, int  size) {
    float h = 0;
    ofstream output;
    ostringstream filename;
    filename << "/home/dj-lawton/Documents/Junior Sophister/Learning_C++/values_for_T"<< T_value <<"_hvary.csv";
    output <<"h,"<<"Energy,"<<"Magnetisation,"<<"Specific Heat,"<<"Magnetic Susceptibility"<<endl;
    for (int i=0;i<num_data_points;i++) {
        h += (h_max/num_data_points);
        IsingModel model(size, T_value, h);
        model.seq_sim(25000);
        output << h<<"," << model.EnergyExpValue()<<","  << model.MagExpValue() <<"," << model.specific_heat() <<"," << model.magnetic_susceptibility() << endl;
    }
    output.close();
}

void Const_h_Varying_T(float h_value, int num_data_points, float T_max, int size) {
    float T = 0;
    ofstream output;
    ostringstream filename;
    filename << "/home/dj-lawton/Documents/Junior Sophister/Learning_C++/values_for_h"<< h_value <<"_Tvary.csv";
    output <<"T,"<<"Energy,"<<"Magnetisation,"<<"Specific Heat,"<<"Magnetic Susceptibility"<<endl;
    for (int i=0;i<num_data_points;i++) {
        T += (T_max/num_data_points);
        IsingModel model(size, T, h_value);
        model.seq_sim(25000);
        output << T<<"," << model.EnergyExpValue()<<","  << model.MagExpValue() <<"," << model.specific_heat() <<"," << model.magnetic_susceptibility() << endl;
    }
    output.close();
}

int main() {
    IsingModel model(10, 1.0, 1.0);

    int num_sweeps_seq = model.conv_seq_sim(35000, 4);
    cout<<"Number of Sweeps to Convergence of Sequential Method:" << num_sweeps_seq << endl;

    // model.lattice_init();
    // int num_sweeps_rand = model.conv_rand_sim(20000, 4);
    // cout<<"Number of Sweeps to Convergence of Random Method: "<< num_sweeps_rand << endl;

    for (float i; i<=6; i++) {

    }
}