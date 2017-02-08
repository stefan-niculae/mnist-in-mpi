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

void print_dimensions(const Matrix& m) {
    cout << m.n_rows << " x " << m.n_cols << endl;
}

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
void add_to_each(const Matrix& matrix, const Matrix& to_add, Matrix& result) {
    if (matrix.n_cols != to_add.n_cols)
        throw runtime_error(string_format("Add to each: matrix width and vector size are different: "
                                                  "%d, %d", matrix.n_rows, matrix.n_rows));
    if (to_add.n_rows != 1)
        throw runtime_error(string_format("Add to each: vector to add has multiple rows: "
                                                  "%d", to_add.n_rows));

    for (int r = 0; r < matrix.n_rows; ++r)
        for (int c = 0; c < matrix.n_cols; ++c)
            result.data[r][c] = matrix.data[r][c] + to_add.data[0][c];
}

void col_wise_sums(const Matrix& matrix, Matrix& result) {
    result.clear();
    for (int r = 0; r < matrix.n_rows; ++r) {
        for (int c = 0; c < matrix.n_cols; ++c)
            result.data[0][c] += matrix.data[r][c];
    }
}

/*** Operations ***/
inline void take_chunk(const Matrix& from, int start_row, Matrix& into) {
    into.data = &(from.data[start_row]);
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
void add(const Matrix& lhs, const Matrix& rhs, Matrix& result) {
    if ((lhs.n_rows != rhs.n_rows) || (lhs.n_cols != rhs.n_cols))
        throw runtime_error(string_format("Matrix addition: number of rows/cols is different: "
                                                  "lhs = (%d, %d), rhs = (%d, %d)",
        lhs.n_rows, lhs.n_cols, rhs.n_rows, rhs.n_cols));

    for (int i = 0; i < rhs.n_rows; ++i)
        for (int j = 0; j < rhs.n_cols; ++j)
            result.data[i][j] = rhs.data[i][j] + lhs.data[i][j];
}

void sub(const Matrix& lhs, const Matrix& rhs, Matrix& result) {
    if ((lhs.n_rows != rhs.n_rows) || (lhs.n_cols != rhs.n_cols))
        throw runtime_error(string_format("Matrix subtraction: number of rows/cols is different: "
                                                  "lhs = (%d, %d), rhs = (%d, %d)",
                                          lhs.n_rows, lhs.n_cols, rhs.n_rows, rhs.n_cols));

    for (int i = 0; i < rhs.n_rows; ++i)
        for (int j = 0; j < rhs.n_cols; ++j)
            result.data[i][j] = rhs.data[i][j] - lhs.data[i][j];
}

void sub_from(Matrix& from, const Matrix& to_sub) {
    // from = from - to_sub
    if ((from.n_rows != to_sub.n_rows) || (from.n_cols != to_sub.n_cols))
        throw runtime_error(string_format("Matrix FROM subtraction: number of rows/cols is different: "
                                                  "from = (%d, %d), to_sub = (%d, %d)",
                                          from.n_rows, from.n_cols, to_sub.n_rows, to_sub.n_cols));

    for (int i = 0; i < from.n_rows; ++i)
        for (int j = 0; j < from.n_cols; ++j)
            from.data[i][j] -= to_sub.data[i][j];
}


// matrix - vector addition
void add_vect (const Matrix& matrix, const Matrix& vect, Matrix& result) {
    if (matrix.n_cols != vect.n_cols)
        throw runtime_error(string_format("Matrix - vector addition: number of rows is different: "
                                                  "%d, %d", matrix.n_cols, vect.n_cols));
    if (vect.n_rows != 1)
        throw runtime_error(string_format("Matrix - vector addition: vector dims: "
                                                  "%d, %d", vect.n_rows, vect.n_cols));
    for (int i = 0; i < matrix.n_rows; ++i)
        for (int j = 0; j < matrix.n_cols; ++j)
            result.data[i][j] = matrix.data[i][j] + vect.data[0][j];
}

// matrix-scalar multiplication
void scalar_mult(double scalar, const Matrix& matrix, Matrix& result) {
    for (int i = 0; i < matrix.n_rows; ++i) {
        for (int j = 0; j < matrix.n_cols; ++j) {
            result.data[i][j] = scalar * matrix.data[i][j];
        }
    }
}

// matrix multiplication
void dot(const Matrix& lhs, const Matrix& rhs, Matrix& result) {
    if (lhs.n_cols != rhs.n_rows)
        throw runtime_error(string_format("Dimensions do not agree for matrix multiplication: "
                      "lhs cols = %d, rhs rows = %d", lhs.n_cols, rhs.n_rows));

    int n = lhs.n_rows;
    int p = lhs.n_cols;
    int m = rhs.n_cols;
    result.clear();
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < p; ++k)
            for (int j = 0; j < m; ++j)
                result.data[i][j] += lhs.data[i][k] * rhs.data[k][j];
}

// element-wise matrix multiplication
void hadamard (const Matrix& lhs, const Matrix& rhs, Matrix& result) {
    if ((lhs.n_rows != rhs.n_rows) || (lhs.n_cols != rhs.n_cols))
        throw runtime_error(string_format("Matrix addition: number of rows/cols is different: "
                                                  "lhs = (%d, %d), rhs = (%d, %d)",
                                          lhs.n_rows, lhs.n_cols, rhs.n_rows, rhs.n_cols));

    for (int i = 0; i < rhs.n_rows; ++i)
        for (int j = 0; j < rhs.n_cols; ++j)
            result.data[i][j] = rhs.data[i][j] * lhs.data[i][j];
}