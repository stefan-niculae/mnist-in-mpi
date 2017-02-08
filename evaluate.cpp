#pragma once

#include <iostream>
#include "matrix.cpp"

double accuracy(const vector<int>& y_true, const vector<int>& y_pred) {
    int n = y_true.size();
    if (y_pred.size() != n)
        throw runtime_error("accuracy: bad dimensions");

    int correct_preds = 0;
    for (int i = 0; i < n; ++i)
        if (y_true[i] == y_pred[i])
            correct_preds++;

    return double(correct_preds) / n;
}

vector<int> labels_from_one_hot(const Matrix& matrix) {
    vector<int> result(matrix.n_rows, 0);

    for (int row = 0; row < matrix.n_rows; ++row)
        for (int col = 0; col < matrix.n_cols; ++col) // {0 0 1 0 0} ~> 2
            if (matrix.data[row][col] == 1) {
                result[row] = col;
                break;
            }

    return result;
}

vector<int> argmax(const Matrix& matrix) {
    vector<int> result(matrix.n_rows, 0);
    double row_max;
    int col_of_max;

    for (int row = 0; row < matrix.n_rows; ++row) {
        double row_max = matrix.data[row][0];
        for (int col = 1; col < matrix.n_cols; ++col) // {0 0 1 0 0} ~> 2
            if (matrix.data[row][col] > row_max) {
                row_max = matrix.data[row][col];
                col_of_max = col;
            }
        result[row] = col_of_max;
    }

    return result;
}
