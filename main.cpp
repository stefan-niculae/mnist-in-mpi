#include <iostream>
#include <vector>
// #include "nn.cpp"




#include <vector>
#include <algorithm>  // max_element
#include <math.h>  // exp

using namespace std;

vector<double> softmax(vector<double> v) {
    // Subtract maximum to avoid overflow
    double max = *max_element(v.begin(), v.end());

    vector<double> expd = v; // copy, does not mutate v
    double sum = 0;
    for (double &x : expd) {
        x = exp(x - max);
        sum += x;
    }

    vector<double> result;
    for (double x : expd) {
        result.push_back(x / sum);
    }

    return result;
}





// using namespace std;

template <class T>
void print(vector<T> v) {
    for (auto x : v)
    {
        cout << x << ' ';
    }
    cout << endl;
}

int main() {
    vector<double> v = {1, 2, 3, 4, 1, 2, 3};
    cout << "v ";
    print(v);

    auto softmaxed = softmax(v);
    cout << "v ";
    print(v);
    cout << "m ";
    print(softmaxed);
}