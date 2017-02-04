#include <iostream>
#include <vector>
// #include "nn.cpp"




#include <vector>
#include <algorithm>  // max_element
#include <math.h>  // exp

using namespace std;

template <class T>
vector<T> softmax(vector<T> v) {
    // Subtract maximum to avoid overflow
    double max = *max_element(v.begin(), v.end());

    auto expd = v; // copy, does not mutate v
    double sum = 0;
    for (double &x : expd) {
        x = exp(x - max);
        sum += x;
    }

    vector<T> result;
    for (double x : expd) {
        result.push_back(x / sum);
    }

    return result;
}

template <class T>
vector<vector<T>> softmax(vector<vector<T>> m) {
    auto result = m;
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





template <class T>
vector<vector<T>> operator* (double scalar, vector<vector<T>> matrix) {
    auto result = matrix;
    for (auto &row : result)
        for (auto& elem : row)
            elem *= scalar;
    return result;
}

template <class T>
int n_rows(vector<vector<T>> m) {
    return m.size();
}

template <class T>
int n_cols(vector<vector<T>> m) {
    return n_rows(m) == 0 ?
        0 :
        m[0].size();
}

template <class T>
vector<vector<T>> operator- (vector<vector<T>> lhs, vector<vector<T>> rhs) {
    auto result = lhs;
    for (int i = 0; i < n_rows(rhs); ++i)
        for (int j = 0; j < n_cols(rhs); ++j)
            result[i][j] -= rhs[i][j];
    return result;
}



int main() {
    vector<double> v0 = {0, 2, 3, 4, 1, 2, 3};
    vector<double> v1 = {1, 5, 6, 2, 6, 7, 8};
    vector<double> v2 = {2, 0, 1, 1, 3, 3, 1};
    vector<double> v3 = {3, 1, 6, 7, 8, 9, 1};

    vector<vector<double>> m = {v0, v1, v2, v3};

    print(m - 2 * m);


    // cout << "v ";
    // print(v);

    // auto softmaxed = softmax(v);
    // cout << "v ";
    // print(v);
    // cout << "m ";
    // print(softmaxed);
}