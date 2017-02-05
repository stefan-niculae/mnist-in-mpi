#pragma once

#include "matrix.cpp"

using namespace std;


/*** Requisites ***/
vector<double> softmax(const vector<double>& v) {
    // Subtract maximum to avoid overflow
    double max = *max_element(v.begin(), v.end());

    vector<double> expd = v; // copy, does not mutate v
    double sum = 0;
    for (double& x : expd) {
        x = exp(x - max);
        sum += x;
    }

    vector<double> result(v.size(), 0.);
    for (int i = 0; i < v.size(); ++i)
        result[i] = expd[i] / sum;

    return result;
}
Matrix softmax(const Matrix& m) {
    Matrix result = m;
    for (auto& row : result)
        row = softmax(row);
    return result;
}

double cross_entropy(const Matrix& Y, const Matrix& Y_prob) {
    // Cost function
    Matrix result = hadamard(Y, log(Y_prob));
    return 1/double(n_rows(Y)) * sum(result);
}




class NeuralNetwork {

    Matrix W;
    Matrix b;
    int n_classes;
    int data_dim;

public:

    NeuralNetwork(int n_classes=10, int data_dim=784) : n_classes(n_classes), data_dim(data_dim) {
        b = blank_matrix(1, n_classes, 0.);
        W = blank_matrix(data_dim, n_classes, 0.);
    }

    double grad(Matrix X, Matrix Y, Matrix &grad_W, Matrix &grad_b) {
        Matrix Y_prob = softmax(add_to_each(X * W, b));
        // error at last layer
        Matrix delta = Y_prob - Y;

        // return gradient of cross entropy cost
        grad_W = transpose(X) * delta;
        Matrix ones = blank_matrix(1, delta.size(), 1.);
        grad_b = ones * delta;

//        cout << n_rows(Y) << " " << n_cols(Y_prob);
//        print(Y_prob);

        return cross_entropy(Y, Y_prob);
    }

    vector<double> train(Matrix X, Matrix Y, int epochs=10, int batch_size=100, double lr=0.1) {
        vector<double> cost_history;

        Matrix grad_W = blank_matrix(data_dim, n_classes, 0.);
        Matrix grad_b = blank_matrix(1, n_classes, 0.);

        for (int epoch=0; epoch<epochs; epoch++) {
            for (int i=0; i<n_rows(X); i+=batch_size) {

                auto cost = grad(chunk(X, i, i+batch_size), chunk(Y, i, i+batch_size), grad_W, grad_b);
                cout << "epoch " << epoch << ", cost = " << cost << endl;
                W = W - lr * 1/batch_size * grad_W;
                b = b - lr * 1/batch_size * grad_b;

                cost_history.push_back(cost);
            }
        }
        return cost_history;
    }

};
