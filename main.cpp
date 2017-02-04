#include <iostream>
#include <vector>
#include "utils.cpp"

using namespace std;


int main() {
    vector<double> v0 = {0, 2, 3, 4, 1, 2, 3};
    vector<double> v1 = {1, 5, 6, 2, 6, 7, 8};
    vector<double> v2 = {2, 0, 1, 1, 3, 3, 1};
    vector<double> v3 = {3, 1, 6, 7, 8, 9, 1};

    vector<vector<double>> m = {v0, v1, v2, v3};

    print(m - 2.0 * m);
}
