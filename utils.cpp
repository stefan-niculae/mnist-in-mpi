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



template <class T>
vector<T> chunk(vector<T> m, int from, int to) {
    return vector<T>(&m[from], &m[to + 1]);;
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
vector<vector<T>> init(int rows, int cols, T value) {
    vector<T> blank_row(cols, value);
    return vector<vector<T>>(rows, blank_row);
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
vector<vector<T>> operator- (vector<vector<T>> lhs, vector<vector<T>> rhs) {
    auto result = lhs;
    for (int i = 0; i < n_rows(rhs); ++i)
        for (int j = 0; j < n_cols(rhs); ++j)
            result[i][j] -= rhs[i][j];
    return result;
}

template <class T>
vector<vector<T>> operator* (vector<vector<T>> lhs, vector<vector<T>> rhs) {
    if (n_cols(lhs) != n_rows(rhs))
        throw "Dimensions do not agree for matrix multiplication!";

    int n = n_rows(lhs);
    int p = n_cols(lhs);
    int m = n_cols(rhs);
    auto result = init(n, m, 0);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < p; ++k)
                result[i][j] += lhs[i][k] * rhs[k][j];

    return result;
}




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
