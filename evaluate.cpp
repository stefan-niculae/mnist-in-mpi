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

// row-wise from_one_hot
vector<int> from_one_hot_matrix(const Matrix& matrix) {
    vector<int> result;
    result.reserve(n_rows(matrix));

    for (const auto& row : matrix)
        result.push_back(from_one_hot(row));

    return result;
}

// row-wise argmax
vector<int> argmax_matrix(const Matrix& matrix) {
    vector<int> result;
    result.reserve(n_rows(matrix));

    for (const auto& row : matrix)
        result.push_back(argmax(row));

    return result;
}
