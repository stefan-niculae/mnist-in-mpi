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

double cross_entropy(const Matrix& Y, const Matrix& Y_prob) {
    // Cost function
    Matrix result = hadamard(Y, log(Y_prob));
    return 1/Y.size() * sum(result);
}




class NeuralNetwork {

    Matrix W;
    Vector b;

public:

    NeuralNetwork(int num_classes=2, int data_dim=3) {
        // b = new Vector(num_classes);
        W = blank_matrix(data_dim, num_classes, 0.);
    }

    double grad(Matrix X, Matrix Y, Matrix &grad_W, Vector &grad_b) {
        Matrix Y_prob = softmax(X * W);
        // error at last layer
        Matrix delta = Y_prob - Y;

        // return gradient of cros entropy cost
        grad_W = delta * X;
        // grad_b = delta;
        return cross_entropy(Y, Y_prob);;
    }

    Vector train(Matrix X, Matrix Y, int epochs=10, int batch_size=100, double lr=0.1) {
        Vector cost_history;
        // X = transpose(X);
//        print(X);
        Y = transpose(Y);
        Matrix grad_W = blank_matrix(3, 2, 0.);
        Vector grad_b;

        for(int epoch=0; epoch<epochs; epoch++) {
            for(int i=0; i<n_rows(X); i+=batch_size) {

                auto cost = grad(chunk(X, i, i+batch_size), chunk(Y, i, i+batch_size), grad_W, grad_b);

                W = W - lr * 1/batch_size * grad_W;
                // b = b - lr * 1/batch_size * grad_b;

                cost_history.push_back(cost);
            }
        }

        return cost_history;
    }

};
