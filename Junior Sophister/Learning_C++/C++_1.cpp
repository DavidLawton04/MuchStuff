# include <iostream>
# include <cmath>

using namespace std;

// Newton's Method

double sqrt(double n) {

    double root = n / 2;
    double *root_ptr = &root;

    for (int i=0; i<10; i++) {
        root = 0.5 * (root+n/root);
    
        cout << "Root of " << n << " is " << *root_ptr << endl;
    }

    return *root_ptr;
}

int main() {

    double n;
    cout << "Enter a number: ";
    cin >> n;

    sqrt(n);

    return 0;
}

