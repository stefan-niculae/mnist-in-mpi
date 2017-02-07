#pragma once

#include <random> // normal_distribution
#include <iostream>
#include <fstream>
#include "matrix.cpp"
#include "evaluate.cpp"


using namespace std;


/*** Requisites ***/
void softmax(const Matrix& matrix, Matrix& result) {
    for (int row = 0; row < matrix.n_rows; ++row) {
        double row_max = matrix[row][0];
        for (int i = 1; i < n; ++i)
            row_max = matrix[row][i] > row_max ? matrix[row][i] : row_max;

        double row_sum = 0;
        // Subtract maximum to avoid overflow
        for (int i = 0; i < n; ++i) {
            result[row][i] = exp(matrix[row][i] - row_max);
            row_sum += result[row][i];
        }

        for (int i = 0; i < n; ++i)
            result[row][i] /= row_sum;
    }
}

// TODO
//double cross_entropy(const Matrix& Y, const Matrix& Y_prob) {
//    // Cost function
//    Matrix result = hadamard(Y, log(Y_prob));
//    return 1/double(n_rows(Y)) * sum(result);
//}

void random_init(Matrix& W, double mean=0., double std=1.) {
    default_random_engine generator;
    normal_distribution<double> distribution(mean, std);

    for (int i = 0; i < W.n_rows; ++i)
        for (int j = 0; j < W.n_cols; ++j)
            W.data[i][j] = distribution(generator);
}




class NeuralNetwork {

    const int N_CLASSES = 10; // 10 digits from 0 to 9
    const int DATA_DIM = 784; // 28*28 = 784 pixels for an image
    const int BATCH_SIZE = 100; // after how many pictures the weights are updated
    const int N_WORKERS = 4; // how many processes compute gradient in parallel
    const int CHUNK_SIZE = BATCH_SIZE / N_WORKERS; // how many rows each worker gets

    Matrix W = Matrix(DATA_DIM, N_CLASSES);
    Matrix b = Matrix(1, N_CLASSES);

    Matrix Y_prob;
    Matrix delta;
    Matrix grad_W;
    Matrix grad_b;

    Matirx XW;
    Matrix XWb;
    Matrix Y_prob;

    Matrix delta;

    Matrix trX;
    Matrix trXdelta;
    Matrix grad_W;

    Matrix delta_sums;

    Matrix chunk_X = Matrix(CHUNK_SIZE, DATA_DIM);
    Matrix chunk_Y = Matrix(CHUNK_SIZE, N_CLASSES);

    Matrix lr_grad_W;
    matrix lr_grad_b;


public:

    NeuralNetwork(int n_classes=10, int data_dim=784)
            : n_classes(n_classes), data_dim(data_dim) {
        // TODO? more layers
    }

    NeuralNetwork(string path) {
        this->load(path);
    }

    void grad(const Matrix& X, const Matrix& Y) {
        dot(X, W, XW); // XW = X * W
        add_to_each(XW, b, XWb); // XWb = X * W + b
        softmax(XWb, Y_prob); // Y_prob = softmax(X * W + b)
        sub(Y_prob, Y, delta); // delta = Y_prob - Y

        double contribution = 1/double(X.n_rows);

        transpose(X, trX); // trX = transpose(X)
        dot(trX, delta, trXdelta); // trXdelta = transpose(X) * delta
        scalar_mult(contribution, trXdelta, grad_W); // grad_W = 1/n * transpose(X) * delta

        col_wise_sums(delta, delta_sums);
        scalar_mult(contribution, delta_sums, grad_b); // grad_b = 1/n * col_wise_sums(delta)

        // TODO maybe
//        return cross_entropy(Y, Y_prob); // cost
    }

    void train(const Matrix& X, const Matrix& Y,
               vector<double>& cost_history, vector<double>& accuracy_history,
               int n_epochs=10, int batch_size=100, double lr=0.1,
               bool verbose=true) {
//        double cost, acc;
//        auto Y_labels = from_one_hot_matrix(Y); // {5, 2, 9, ... }

        random_init(W);

        for (int epoch = 1; epoch <= n_epochs; ++epoch) {
            if (verbose)
                cout << string_format("Epoch %d / %d", epoch, n_epochs) << endl;

            for (int batch_start = 0; batch_start < X.n_rows; batch_start += BATCH_SIZE) {
                // TODO? make random batch generator
                take_chunk(X, batch_start, chunk_X); // chunk_X = X[batch_start ... batch_start + CHUNK_SIZE]
                take_chunk(Y, batch_start, chunk_Y);
                grad(chunk_X, chunk_Y); // grad_W and grad_b are now filled with result

                // TODO? regularization
                scalar_mult(-lr, grad_W, lr_grad_W);
                scalar_mult(-lr, grad_b, lr_grad_b);
                sub_from(W, lr_grad_W); // W = W - lr * grad_W
                sub_from(b, lr_grad_b); // b = b - lr * grad_b

//                cost_history.push_back(cost); // TODO
            }

            // TODO: add early stopping
            // Compute accuracy after each epoch
//            acc = accuracy(predict(X), Y_labels); // TODO
//            accuracy_history.push_back(acc);
        }
    }

//    vector<int> predict(const Matrix& X) {
//        Matrix Y_prob = softmax(X * W + b);
//        return argmax_matrix(Y_prob);
//    }
//
//    int predict_one(const vector<double>& pixels, double& confidence) {
//        // TODO: refactor to use .predict()
//        Matrix X = {pixels};
//        vector<double> y_prob = softmax(X * W + b)[0]; // just one image, take the first row
//
//        int digit_predicted = argmax(y_prob);
//        confidence = y_prob[digit_predicted];
//        return digit_predicted;
//    }


    /*** Serialization ***/
    void save(string path) {
        ofstream file(path);
        if (!file.is_open())
            throw runtime_error(string_format("Could not open for model save: " + path));

        file << *this;

        file.close();
    }

    void load(string path) {
        ifstream file(path);
        if (!file.is_open())
            throw runtime_error(string_format("Could not open for model load: " + path));

        file >> *this;

        file.close();
    }

    friend ostream& operator<<(ostream& os, const NeuralNetwork& net);
    friend istream& operator>>(istream& is, NeuralNetwork& net);
};


ostream& operator<<(ostream& os, const NeuralNetwork& net) {
    os << net.n_classes << ' ' << net.data_dim << endl;
    os << net.b << net.W;  // place W last so it can be read until end of file
    return os;
}

istream& operator>>(istream& is, NeuralNetwork& net) {
    is >> net.n_classes >> net.data_dim;

    // net.b is a 1xn matrix, if we read a matrix normally, we read until end of file,
    // but a vector is read until end of file
    vector<double> b_vect;
    do {
        is >> b_vect;
    } while (b_vect.size() == 0);  // to account for cursor left at end of line
    net.b = {b_vect};

    is >> net.W;
    return is;
}