#pragma once

#include <vector>

using namespace std;


// vector-wise softmax
template <class T>
vector<T> softmax(vector<T> v);

// matrix-wise softmax
template <class T>
vector<vector<T>> softmax(vector<vector<T>> m);


// submatrix
template <class T>
vector<T> chunk(vector<T> m, int from, int to);

// dimension 0
template <class T>
int n_rows(vector<vector<T>> m);

// dimension 1
template <class T>
int n_cols(vector<vector<T>> m);


// scalar - matrix multiplication
template <class T>
vector<vector<T>> operator*(double scalar, vector<vector<T>> matrix);

// matrix subtraction
template <class T>
vector<vector<T>> operator-(vector<vector<T>> lhs, vector<vector<T>> rhs);


// print vector
template <class T>
void print(vector<T> v);

// print matrix
template <class T>
void print(vector<vector<T>> m);