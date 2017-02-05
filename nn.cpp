#pragma once

#include "matrix.cpp"

using namespace std;


/*** Requisites ***/
Vector softmax(const Vector& v) {
    // Subtract maximum to avoid overflow
    double max = *max_element(v.begin(), v.end());

    auto expd = Vector(v); // copy, does not mutate v
    double sum = 0;
    for (double& x : expd) {
        x = exp(x - max);
        sum += x;
    }

    Vector result(v.size(), 0.);
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
    return 1/Y.size() * sum(result);
}




class NeuralNetwork {

    Matrix W;
    Vector b;
    int n_classes;
    int data_dim;

public:

    NeuralNetwork(int n_classes=10, int data_dim=784) : n_classes(n_classes), data_dim(data_dim) {
        b = Vector(n_classes);
        W = blank_matrix(data_dim, n_classes, 0.);
    }

    double grad(Matrix X, Matrix Y, Matrix &grad_W, Vector &grad_b) {
//        cout << "X dims: " << n_rows(X) << " " << n_cols(X) << endl;
//        cout << "W dims: " << n_rows(W) << " " << n_cols(W) << endl;

//        cout << "Result: " << n_rows(X * W + b) << " " << n_cols(X * W + b) << endl;

        Matrix Y_prob = softmax(X * W + b);

//        cout << "Result Softmax: " << n_rows(Y_prob) << " " << n_cols(Y_prob) << endl;

        // error at last layer
        Matrix delta = Y_prob - Y;

        // return gradient of cross entropy cost
        grad_W = transpose(X) * delta;
        // grad_b = delta;
        return cross_entropy(Y, Y_prob);
    }

    Vector train(Matrix X, Matrix Y, int epochs=10, int batch_size=100, double lr=0.1) {
        Vector cost_history;

        Matrix grad_W = blank_matrix(this->data_dim, this->n_classes, 0.);
        Vector grad_b;

        for (int epoch=0; epoch<epochs; epoch++) {
            for (int i=0; i<n_rows(X); i+=batch_size) {

                auto cost = grad(chunk(X, i, i+batch_size), chunk(Y, i, i+batch_size), grad_W, grad_b);
                cout << "epoch " << epoch << ", cost = " << cost << endl;
                W = W - lr * 1/batch_size * grad_W;
                // b = b - lr * 1/batch_size * grad_b;

                cost_history.push_back(cost);
            }
        }
        return cost_history;
    }

};
