#pragma once

#include <random> // normal_distribution
#include <iostream>
#include <fstream>
#include "matrix.cpp"
//#include "evaluate.cpp" // TODO


using namespace std;


/*** Requisites ***/
void softmax(const Matrix& matrix, Matrix& result) {
    for (int row = 0; row < matrix.n_rows; ++row) {
        double row_max = matrix.data[row][0];
        for (int i = 1; i < matrix.n_cols; ++i)
            row_max = matrix.data[row][i] > row_max ? matrix.data[row][i] : row_max;

        double row_sum = 0;
        // Subtract maximum to avoid overflow
        for (int i = 0; i < matrix.n_cols; ++i) {
            result.data[row][i] = exp(matrix.data[row][i] - row_max);
            row_sum += result.data[row][i];
        }

        for (int i = 0; i < matrix.n_cols; ++i)
            result.data[row][i] /= row_sum;
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

    const int N_SAMPLES = 250;
    const int N_CLASSES = 10; // 10 digits from 0 to 9
    const int DATA_DIM = 784; // 28*28 = 784 pixels for an image
    const int BATCH_SIZE = 100; // after how many pictures the weights are updated
    const int N_WORKERS = 4; // how many processes compute gradient in parallel
    const int CHUNK_SIZE = BATCH_SIZE / N_WORKERS; // how many rows each worker gets

    // X dimensions: N_SAMPLES x DATA_DIM
    // Y dimensions: N_SAMPLES x N_CLASSES

    Matrix W = Matrix(DATA_DIM, N_CLASSES);        // TODO? more layers
    Matrix b = Matrix(1, N_CLASSES);

    Matrix chunk_X = Matrix(CHUNK_SIZE, DATA_DIM);
    Matrix chunk_Y = Matrix(CHUNK_SIZE, N_CLASSES);
    // chunk sized
        Matrix XW = Matrix(CHUNK_SIZE, N_CLASSES); // chunk_X * W ie (CHUNK_SIZE, DATA_DIM) * (DATA_DIM, N_CLASSES)
        Matrix XWb = Matrix(CHUNK_SIZE, N_CLASSES);
        Matrix Y_prob = Matrix(CHUNK_SIZE, N_CLASSES);
        Matrix delta = Matrix(CHUNK_SIZE, N_CLASSES);

        Matrix trX = Matrix(DATA_DIM, CHUNK_SIZE); // transpose CHUNK_SIZE, DATA_DIM
        Matrix trXdelta = Matrix(DATA_DIM, N_CLASSES); // DATA_DIM, CHUNK_SIZE * CHUNK_SIZE, N_CLASSES

        Matrix delta_sums = Matrix(1, N_CLASSES); // like b


    Matrix lr_grad_W = Matrix(DATA_DIM, N_CLASSES); // like W
    Matrix lr_grad_b = Matrix(1, N_CLASSES); // like b
    Matrix grad_W = Matrix(DATA_DIM, N_CLASSES); // like W
    Matrix grad_b = Matrix(1, N_CLASSES); // like b


public:

    NeuralNetwork() { }

//    NeuralNetwork(string path) {
//        this->load(path);
//    }

    void grad() {
        print_dimensions(chunk_X);
        print_dimensions(W);
        print_dimensions(XW);
        dot(chunk_X, W, XW); // XW = X * W
        cout << "after dot1\n";
        add_to_each(XW, b, XWb); // XWb = X * W + b
//        cout << "after add_to_each/**/\n";
        softmax(XWb, Y_prob); // Y_prob = softmax(X * W + b)
//        cout << "after softmax\n";
        sub(Y_prob, chunk_Y, delta); // delta = Y_prob - Y
//        cout << "after sub1\n";

        double contribution = 1/double(CHUNK_SIZE);

        transpose(chunk_X, trX); // trX = transpose(X)
//        cout << "after transpose\n";
        dot(trX, delta, trXdelta); // trXdelta = transpose(X) * delta
//        cout << "after dot2\n";
        scalar_mult(contribution, trXdelta, grad_W); // grad_W = 1/n * transpose(chunk_X) * delta

        col_wise_sums(delta, delta_sums);
        scalar_mult(contribution, delta_sums, grad_b); // grad_b = 1/n * col_wise_sums(delta)

        // TODO maybe
//        return cross_entropy(Y, Y_prob); // cost
    }

    void train(const Matrix& X, const Matrix& Y,
//               vector<double>& cost_history, vector<double>& accuracy_history,
               int n_epochs=10, int batch_size=100, double lr=0.1,
               bool verbose=true) {
//        double cost, acc;
//        auto Y_labels = from_one_hot_matrix(Y); // {5, 2, 9, ... }

        random_init(W);
//        cout << "after init W" << endl;

        for (int epoch = 1; epoch <= n_epochs; ++epoch) {
            if (verbose)
                cout << string_format("Epoch %d / %d", epoch, n_epochs) << endl;

            for (int batch_start = 0; batch_start < X.n_rows; batch_start += BATCH_SIZE) {
                // TODO? make random batch generator
                take_chunk(X, batch_start, chunk_X); // chunk_X = X[batch_start ... batch_start + CHUNK_SIZE]
//                cout << "after take chunk X\n";
                take_chunk(Y, batch_start, chunk_Y);
//                cout << "after take chunk Y\n";
                grad(); // grad_W and grad_b are now filled with result
                cout << "after grad\n";

                // TODO? regularization
                scalar_mult(-lr, grad_W, lr_grad_W);
                scalar_mult(-lr, grad_b, lr_grad_b);
//                cout << "after scalar mults\n";
                sub_from(W, lr_grad_W); // W = W - lr * grad_W
                sub_from(b, lr_grad_b); // b = b - lr * grad_b
                cout << "after update W & b\n";
//                cost_history.push_back(cost); // TODO
            }
            cout << "at end of epoch loop";

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

//    void load(string path) {
//        ifstream file(path);
//        if (!file.is_open())
//            throw runtime_error(string_format("Could not open for model load: " + path));
//
//        file >> *this;
//
//        file.close();
//    }

    friend ostream& operator<<(ostream& os, const NeuralNetwork& net);
    friend istream& operator>>(istream& is, NeuralNetwork& net);
};


ostream& operator<<(ostream& os, const NeuralNetwork& net) {
    os << net.b << net.W;  // place W last so it can be read until end of file
    return os;
}

//istream& operator>>(istream& is, NeuralNetwork& net) {
//    // net.b is a 1xn matrix, if we read a matrix normally, we read until end of file,
//    // but a vector is read until end of file
//    vector<double> b_vect;
//    do {
//        is >> b_vect;
//    } while (b_vect.size() == 0);  // to account for cursor left at end of line
//    net.b = {b_vect};
//
//    is >> net.W;
//    return is;
//}