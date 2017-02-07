#pragma once

#include <math.h>  // exp, math
#include "utils.cpp"
#include <iostream>

class Matrix {

public:
    double** data;
    int n_rows;
    int n_cols;
    bool is_chunk = false;

    Matrix(int n_rows,int n_cols): n_rows(n_rows), n_cols(n_cols) {
        data = new double*[n_rows];
        for (int i = 0; i < n_rows; ++i)
            data[i] = new double[n_cols];
    }

    ~Matrix() {
        if (is_chunk)
            return;
        for (int i = 0; i < n_rows; ++i)
            delete[] data[i];
        delete[] data;
    }

    void clear() {
        for (int i = 0; i < n_rows; ++i)
            for (int j = 0; j < n_cols; ++j)
                data[i][j] = 0;
    }
};



/*** IO ***/
std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    for (int i = 0; i < matrix.n_rows; ++i) {
        for (int j = 0; j < matrix.n_cols; ++j)
            os << matrix.data[i][j] << ' ';
        os << std::endl;
    }
    return os;
}
std::istream& operator>>(std::istream& is, Matrix& matrix) {
    for (int i = 0; i < matrix.n_rows; ++i)
        for (int j = 0; j < matrix.n_cols; ++j)
            is >> matrix.data[i][j];
    return is;
}

/*** Operators ***/
// Adds first element of `to_add` to each element on the first column of `matrix`
// second element of `to_add` to each of second column in `matrix`... etc
void add_to_each(Matrix& matrix, const Matrix& to_add, Matrix& result) {
    if (matrix.n_cols != to_add.n_cols)
        throw runtime_error(string_format("Add to each: matrix width and vector size are different: "
                                                  "%d, %d", matrix.n_rows, matrix.n_rows));
    if (to_add.n_rows != 1)
        throw runtime_error(string_format("Add to each: vector to add has multiple rows: "
                                                  "%d", to_add.n_rows));

    for (int r = 0; r < to_add.n_rows; ++r)
        for (int c = 0; c < matrix.n_cols; ++c)
            result.data[r][c] += to_add.data[0][c];
}

void col_wise_sums(const Matrix& matrix, Matrix& result) {
    result.clear();
    for (int r = 0; r < matrix.n_rows; ++r) {
        for (int c = 0; c < matrix.n_cols; ++c)
            result.data[0][c] += matrix.data[r][c];
    }
}

/*** Operations ***/
inline void take_chunk(const Matrix& from, int start_index, Matrix& into) {
    into.data = from.data + start_index * from.n_cols; // skip n rows, where n is the start index
    into.is_chunk = true;
}

// matrix transposition
void transpose(const Matrix& matrix, Matrix& transposed) {
    for (int j=0; j < matrix.n_cols; j++) {
        for (int i=0; i < matrix.n_rows; i++) {
            transposed.data[j][i] = matrix.data[i][j];
        }
    }
}


// element-wise log
void log(const Matrix& matrix, Matrix& result) {
    for (int i = 0; i < matrix.n_rows; ++i) {
        for (int j = 0; j < matrix.n_cols; ++j) {
            result.data[i][j] = log(matrix.data[i][j]);
        }
    }
}

// whole matrix sum
void sum(const Matrix& matrix, double result) {
    result = 0;
    for (int i = 0; i < matrix.n_rows ; ++i) {
        for (int j = 0; j < matrix.n_cols; ++j) {
            result += matrix.data[i][j];
        }
    }
}

//
///*** Operators ***/
//
//// matrix addition
void add (const Matrix& lhs, const Matrix& rhs, Matrix& result) {
    // If rhs is a row-vector, add it to each row of the lhs
    if (rhs.n_rows == 1 && rhs.n_cols == lhs.n_cols)
//        add_to_each(lhs, rhs, result);

    if ((lhs.n_rows != rhs.n_rows) || (lhs.n_cols != rhs.n_cols))
        throw runtime_error(string_format("Matrix addition: number of rows/cols is different: "
                                                  "lhs = (%d, %d), rhs = (%d, %d)",
        lhs.n_rows, lhs.n_cols, rhs.n_rows, rhs.n_cols));

    for (int i = 0; i < rhs.n_rows; ++i)
        for (int j = 0; j < rhs.n_cols; ++j)
            result.data[i][j] = rhs.data[i][j] + lhs.data[i][j];
}
//
//// matrix - vector addition
//template <class T>
//vector<vector<T>> operator+ (const vector<vector<T>>& matrix, const vector<T>& vect) {
//    if (n_cols(matrix) != vect.size())
//        throw runtime_error(string_format("Matrix - vector addition: number of rows is different: "
//                                                  "%d, %d", n_cols(matrix), vect.size()));
//
//    auto result = matrix;
//    for (int i = 0; i < n_rows(matrix); ++i)
//        for (int j = 0; j < n_cols(matrix); ++j)
//            result[i][j] += vect[j];
//    return result;
//}
//
//
//// matrix subtraction
//template <class T>
//vector<vector<T>> operator- (const vector<vector<T>>& lhs, const vector<vector<T>>& rhs) {
//    if ((n_rows(lhs) != n_rows(rhs)) || n_cols(lhs) != n_cols(rhs))
//        throw runtime_error(string_format("Matrix subtraction: number of rows/cols is different: "
//                                                  "lhs = (%d, %d), rhs = (%d, %d)",
//                                          n_rows(lhs), n_cols(lhs), n_rows(rhs), n_cols(rhs)));
//
//    auto result = lhs;
//    for (int i = 0; i < n_rows(rhs); ++i)
//        for (int j = 0; j < n_cols(rhs); ++j)
//            result[i][j] -= rhs[i][j];
//    return result;
//}
//
//// matrix-scalar multiplication
//template <class T>
//vector<vector<T>> operator* (double scalar, const vector<vector<T>>& matrix) {
//    auto result = matrix;
//    for (auto& row : result)
//        for (auto& elem : row)
//            elem *= scalar;
//    return result;
//}
//
//// TODO: faster method
//// matrix multiplication
//template <class T>
//vector<vector<T>> operator* (const vector<vector<T>>& lhs, const vector<vector<T>>& rhs) {
//    if (n_cols(lhs) != n_rows(rhs))
//        throw runtime_error(string_format("Dimensions do not agree for matrix multiplication: "
//                      "lhs cols = %d, rhs rows = %d", n_cols(lhs), n_rows(rhs)));
//
//    int n = n_rows(lhs);
//    int p = n_cols(lhs);
//    int m = n_cols(rhs);
//    auto result = blank_matrix(n, m, 0.);
//
//    for (int i = 0; i < n; ++i)
//        for (int k = 0; k < p; ++k)
//            for (int j = 0; j < m; ++j)
//                result[i][j] += lhs[i][k] * rhs[k][j];
//
//    return result;
//}
//
//// element-wise matrix multiplication
//template <class T>
//vector<vector<T>> hadamard(const vector<vector<T>>& lhs, const vector<vector<T>>& rhs) {
//    if ((n_rows(lhs) != n_rows(rhs)) || n_cols(lhs) != n_cols(rhs))
//        throw runtime_error(string_format("Matrix element-wise multiplication: number of rows/cols is different: "
//                                                  "lhs = (%d, %d), rhs = (%d, %d)",
//                                          n_rows(lhs), n_cols(lhs), n_rows(rhs), n_cols(rhs)));
//
//    auto result = lhs;
//    for (int i = 0; i < n_rows(rhs); ++i)
//        for (int j = 0; j < n_cols(rhs); ++j)
//            result[i][j] *= rhs[i][j];
//    return result;
//}
