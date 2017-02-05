#pragma once
#include <vector>
#include <algorithm>  // max_element
#include <math.h>  // exp
#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>


using namespace std;

typedef vector<double> Vector;
typedef vector<vector<double>> Matrix;

// TODO: make const& for (almost) every function here

Vector softmax(const Vector& v) {
    // Subtract maximum to avoid overflow
    double max = *max_element(v.begin(), v.end());

    auto expd = Vector(v); // copy, does not mutate v
    double sum = 0;
    for (double &x : expd) {
        x = exp(x - max);
        sum += x;
    }

    Vector result(v.size(), 0.);
    for (double x : expd) {
        result.push_back(x / sum);
    }

    return result;
}

Matrix softmax(const Matrix& m) {
    auto result = Matrix(m);
    for (auto& row : result)
        row = softmax(row);
    return result;
}



template <class T>
vector<T> chunk(vector<T> m, int from, int to) {
    if(from < 0) {
        throw "From < 0";
    }
    if (to > m.size()) to = m.size();
    return vector<T>(&m[from], &m[to]);
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
vector<vector<T>> blank_matrix(int rows, int cols, T value) {
    return vector<vector<T>>(rows, vector<T>(cols, value));
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
    if(n_rows(lhs) != n_rows(rhs)) {
        cout << n_rows(lhs) << " " << n_rows(rhs);
        throw "Number of rows is different"; //, %d != %d", n_rows(lhs), n_rows(rhs));
    } else if(n_cols(lhs) != n_cols(rhs)) {
        throw "Number of colums is different"; //, %d != %d", n_cols(lhs), n_cols(rhs));
    }
    auto result = lhs;
    for (int i = 0; i < n_rows(rhs); ++i)
        for (int j = 0; j < n_cols(rhs); ++j)
            result[i][j] -= rhs[i][j];
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

template <class T>
vector<vector<T>> operator* (vector<vector<T>> lhs, vector<vector<T>> rhs) {
    if (n_cols(lhs) != n_rows(rhs)) {
        print(lhs);
        print(rhs);
        cout << n_cols(lhs) << n_rows(rhs);
        throw "Dimensions do not agree for matrix multiplication!";
    }

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



void print_image(Vector pixels) {
    for (int i = 0; i < 28 * 28; ++i) {
        if (i > 0 && i % 28 == 0)
            cout << endl << endl;

        if (pixels[i] == 0)
            cout << "  ";
        else
            cout << int(pixels[i] * 100); // assumes 0..1
        cout << ' ';
    }
    cout << endl << endl;
}


template <class T>
vector<vector<T>> transpose(vector<vector<T>> matrix) {
    vector<vector<T>> matrix_t(n_cols(matrix), vector<T>(n_rows(matrix)));
    for(int j=0; j<n_cols(matrix); j++) {
        for(int i=0; i<n_rows(matrix); i++) {
            matrix_t[j][i] = matrix[i][j];
        }
    }
    return matrix_t;
}

template <class T>
vector<vector<T>> operator+ (vector<vector<T>> lhs, vector<vector<T>> rhs) {
    if(n_rows(lhs) != n_rows(rhs)) {
        throw sprintf("Number of rows is different, %d != %d", n_rows(lhs), n_rows(rhs));
    } else if(n_cols(lhs) != n_cols(rhs)) {
        throw sprintf("Number of colums is different, %d != %d", n_cols(lhs), n_cols(rhs));
    }
    auto result = lhs;
    for (int i = 0; i < n_rows(rhs); ++i)
        for (int j = 0; j < n_cols(rhs); ++j)
            result[i][j] += rhs[i][j];
    return result;
}

template <class T>
vector<vector<T>> hadamard (vector<vector<T>> lhs, vector<vector<T>> rhs) {
    if(n_rows(lhs) != n_rows(rhs)) {
        throw "Number of rows is different"; //, %d != %d", n_rows(lhs), n_rows(rhs));
    } else if(n_cols(lhs) != n_cols(rhs)) {
        throw "Number of colums is different"; //,%d != %d", n_cols(lhs), n_cols(rhs));
    }
    auto result = lhs;
    for (int i = 0; i < n_rows(rhs); ++i)
        for (int j = 0; j < n_cols(rhs); ++j)
            result[i][j] *= rhs[i][j];
    return result;
}

template <class T>
vector<vector<T>> log (vector<vector<T>> matrix) {
    auto result = matrix;
    for (auto &row : result)
        for (auto& elem : row)
            elem = log(elem);
    return result;
}

template <class T>
T sum (vector<vector<T>> matrix) {
    T result = 0;
    for (auto row : matrix)
        for (auto elem : row)
            result += elem;
    return result;
}

template <class T>
T CE(vector<vector<T>> Y, vector<vector<T>> Y_prob) {
    // Cost function - Cross Entropy
    vector<vector<T>> result = hadamard(Y, log(Y_prob));
    return 1/Y.size() * sum(result);
}

template <class T>
vector<vector<T>> operator+ (vector<vector<T>> matrix, vector<T> vect) {
    if(n_rows(matrix) != n_rows(vect)) {
        throw sprintf("Number of rows is different, %d != %d", n_rows(matrix), n_rows(vect));
    }
    auto result = matrix;
    for (int i = 0; i < n_rows(matrix); ++i)
        for (int j = 0; j < n_cols(matrix); ++j)
            result[i][j] += vect[i];
    return result;
}

template <class T>
vector<T> operator* (double scalar, vector<T> vector) {
    auto result = vector;
    for (auto &elem : result)
            elem *= scalar;
    return result;
}