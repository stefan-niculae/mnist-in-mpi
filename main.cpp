#include <iostream>
#include "utils.cpp"
#include "nn.cpp"

using namespace std;


int main() {
    vector<double> v0 = {0, 2, 3};
    vector<double> v1 = {1, 5, 6};
    vector<double> v2 = {2, 0, 1};
    vector<double> v3 = {3, 4, 1};

    vector<double> y1 = {1, 0};
    vector<double> y2 = {1, 0};
    vector<double> y3 = {0, 1};
    vector<double> y4 = {0, 1};
    
    
    vector<vector<double>> Y = {y1, y2, y3, y4};
    vector<vector<double>> X = {v0, v1, v2, v3};

    NN model(Y[0].size(), X[0].size());

    try {
        auto history = model.train(X, Y, 10, 2, 1);
    }
    catch (const char* err){
        cout << err;
    }
    
    char* s;
    cin >> s;
}
