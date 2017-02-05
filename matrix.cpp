#pragma once

#include <algorithm>  // max_element
#include <math.h>  // exp, math
#include "vector.cpp"
#include "utils.cpp"

typedef vector<Vector> Matrix;


template <class T>
void print(const vector<vector<T>>& m) {
    for (const auto& row : m)
        print(row);
    cout << endl;
}

template <class T>
int n_rows(const vector<vector<T>>& m) {
    return m.size();
}

template <class T>
int n_cols(const vector<vector<T>>& m) {
    return n_rows(m) == 0 ?
           0 :
           m[0].size();
}

template <class T>
vector<T> chunk(const vector<T>& m, int from, int to) {
    if (from < 0)
        throw "From < 0";
    if (to > m.size()) to = m.size();
    return vector<T>(&m[from], &m[to]);
}

template <class T>
inline vector<vector<T>> blank_matrix(int rows, int cols) {
    // Default value
    return vector<vector<T>>(rows, vector<T>(cols));
}
template <class T>
inline vector<vector<T>> blank_matrix(int rows, int cols, T value) {
    return vector<vector<T>>(rows, vector<T>(cols, value));
}




/*** Operations ***/

// matrix transposition
template <class T>
vector<vector<T>> transpose(const vector<vector<T>>& matrix) {
    vector<vector<T>> transposed = vector<vector<T>> (n_cols(matrix), vector<T>(n_rows(matrix)));
    for (int j=0; j<n_cols(matrix); j++) {
        for (int i=0; i<n_rows(matrix); i++) {
            transposed[j][i] = matrix[i][j];
        }
    }
    return transposed;
}

// element-wise log
template <class T>
vector<vector<T>> log(const vector<vector<T>>& matrix) {
    auto result = matrix;
    for (auto& row : result)
        for (auto& elem : row)
            elem = log(elem);
    return result;
}

// whole matrix sum
template <class T>
T sum (const vector<vector<T>>& matrix) {
    T result = 0;
    for (const auto& row : matrix)
        for (const auto& elem : row)
            result += elem;
    return result;
}





/*** Operators ***/

// matrix addition
template <class T>
vector<vector<T>> operator+ (const vector<vector<T>>& lhs, const vector<vector<T>>& rhs) {
    if ((n_rows(lhs) != n_rows(rhs)) || n_cols(lhs) != n_cols(rhs))
        throw runtime_error(string_format("Matrix addition: number of rows/cols is different: "
                                                  "lhs = (%d, %d), rhs = (%d, %d)",
        n_rows(lhs), n_cols(lhs), n_rows(rhs), n_cols(rhs)));

    auto result = lhs;
    for (int i = 0; i < n_rows(rhs); ++i)
        for (int j = 0; j < n_cols(rhs); ++j)
            result[i][j] += rhs[i][j];
    return result;
}

// matrix - vector addition
template <class T>
vector<vector<T>> operator+ (const vector<vector<T>>& matrix, const vector<T>& vect) {
    if (n_cols(matrix) != vect.size())
        throw runtime_error(string_format("Matrix - vector addition: number of rows is different: "
                                                  "%d, %d", n_cols(matrix), vect.size()));

    auto result = matrix;
    for (int i = 0; i < n_rows(matrix); ++i)
        for (int j = 0; j < n_cols(matrix); ++j)
            result[i][j] += vect[j];
    return result;
}


// matrix subtraction
template <class T>
vector<vector<T>> operator- (const vector<vector<T>>& lhs, const vector<vector<T>>& rhs) {
    if ((n_rows(lhs) != n_rows(rhs)) || n_cols(lhs) != n_cols(rhs))
        throw runtime_error(string_format("Matrix subtraction: number of rows/cols is different: "
                                                  "lhs = (%d, %d), rhs = (%d, %d)",
                                          n_rows(lhs), n_cols(lhs), n_rows(rhs), n_cols(rhs)));

    auto result = lhs;
    for (int i = 0; i < n_rows(rhs); ++i)
        for (int j = 0; j < n_cols(rhs); ++j)
            result[i][j] -= rhs[i][j];
    return result;
}

// matrix-scalar multiplication
template <class T>
vector<vector<T>> operator* (double scalar, const vector<vector<T>>& matrix) {
    auto result = matrix;
    for (auto& row : result)
        for (auto& elem : row)
            elem *= scalar;
    return result;
}

// matrix multiplication
template <class T>
vector<vector<T>> operator* (const vector<vector<T>>& lhs, const vector<vector<T>>& rhs) {
    if (n_cols(lhs) != n_rows(rhs))
        throw runtime_error(string_format("Dimensions do not agree for matrix multiplication: "
                      "lhs cols = %d, rhs rows = %d", n_cols(lhs), n_rows(rhs)));

    int n = n_rows(lhs);
    int p = n_cols(lhs);
    int m = n_cols(rhs);
    auto result = blank_matrix(n, m, 0.);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < p; ++k)
                result[i][j] += lhs[i][k] * rhs[k][j];

    return result;
}

// element-wise matrix multiplication
template <class T>
vector<vector<T>> hadamard(const vector<vector<T>>& lhs, const vector<vector<T>>& rhs) {
    if ((n_rows(lhs) != n_rows(rhs)) || n_cols(lhs) != n_cols(rhs))
        throw runtime_error(string_format("Matrix element-wise multiplication: number of rows/cols is different: "
                                                  "lhs = (%d, %d), rhs = (%d, %d)",
                                          n_rows(lhs), n_cols(lhs), n_rows(rhs), n_cols(rhs)));

    auto result = lhs;
    for (int i = 0; i < n_rows(rhs); ++i)
        for (int j = 0; j < n_cols(rhs); ++j)
            result[i][j] *= rhs[i][j];
    return result;
}
