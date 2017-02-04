#include <iostream>
#include <vector>
// #include "nn.cpp"




#include <vector>
#include <algorithm>  // max_element
#include <math.h>  // exp

using namespace std;

typedef vector<double> Vector;
typedef vector<vector<double>> Matrix;

Vector softmax(Vector v) {
    // Subtract maximum to avoid overflow
    double max = *max_element(v.begin(), v.end());

    Vector expd = v; // copy, does not mutate v
    double sum = 0;
    for (double &x : expd) {
        x = exp(x - max);
        sum += x;
    }

    Vector result;
    for (double x : expd) {
        result.push_back(x / sum);
    }

    return result;
}

Matrix softmax(Matrix m) {
    Matrix result = m;
    for (auto& row : m)
        row = softmax(row);
}

// using namespace std;

template <class T>
void print(vector<T> v) {
    for (auto x : v)
        cout << x << ' ';
    cout << endl;
}

template <class T>
void print(vector<vector<T>> m) {
    for (auto row : m)
        print(row);
    cout << endl;
}

template <class T>
vector<T> chunk(vector<T> m, int from, int to) {
    return vector<T>(&m[from], &m[to + 1]);;
}

int main() {
    vector<double> v0 = {0, 2, 3, 4, 1, 2, 3};
    vector<double> v1 = {1, 5, 6, 2, 6, 7, 8};
    vector<double> v2 = {2, 0, 1, 1, 3, 3, 1};
    vector<double> v3 = {3, 1, 6, 7, 8, 9, 1};

    vector<vector<double>> m = {v0, v1, v2, v3};

    print(m);
    int from = 1, to = 2;
    print(chunk(m, from, to));

    // cout << "v ";
    // print(v);

    // auto softmaxed = softmax(v);
    // cout << "v ";
    // print(v);
    // cout << "m ";
    // print(softmaxed);
}