#pragma once

#include <random> // normal_distribution
#include <iostream>
#include <fstream>
#include "matrix.cpp"
#include "evaluate.cpp"
#include "mpi.h"


using namespace std;


/*** MPI constants ***/
const int MASTER = 0;
const int GRAD_W_TAG    = 0;
const int GRAD_B_TAG    = 1;
//const int UPDATED_W_TAG = 2;



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

Matrix random_init(int rows, int cols, double mean=0., double std=1.) {
    default_random_engine generator;
    normal_distribution<double> distribution(mean, std);

    Matrix result = blank_matrix(rows, cols, 0.);
    for (auto& row : result)
        for (double& elem : row)
            elem = distribution(generator);

    return result;
}




class NeuralNetwork {

    int n_classes;
    int data_dim;
    Matrix W;
    Matrix b;

public:

    NeuralNetwork(int n_classes=10, int data_dim=784)
            : n_classes(n_classes), data_dim(data_dim) {
        b = blank_matrix(1, n_classes, 0.);
        W = random_init(data_dim, n_classes);
        // TODO? more layers
    }

    NeuralNetwork(string path) {
        this->load(path);
    }

    // TODO: parallelization
    double grad(Matrix X, Matrix Y,
                Matrix &grad_W, Matrix &grad_b) {
        double n = n_rows(X);  // double instead of int so division works

        Matrix Y_prob = softmax(X * W + b); // add bias to each
        // Error at last layer
        Matrix delta = Y_prob - Y;

        // Gradient of cross entropy cost
        grad_W = 1/n * transpose(X) * delta;
        grad_b = 1/n * col_wise_sums(delta);

        return cross_entropy(Y, Y_prob); // cost
    }

    void train(Matrix X, Matrix Y,
               vector<double>& cost_history, vector<double>& accuracy_history,
               int n_epochs=10, int batch_size=100, double lr=0.1,
               bool verbose=true) {
        // MPI initialization
        MPI_Init(NULL, NULL);
        int rank, n_processes;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_processes);
        int n_workers = n_processes - 1; // first process is the master
        int chunk_size = batch_size / n_workers;


        double cost, acc;
        auto Y_labels = from_one_hot_matrix(Y); // {5, 2, 9, ... }

        Matrix total_grad_W = blank_matrix(data_dim, n_classes, 0.);
        Matrix total_grad_b = blank_matrix(1, n_classes, 0.);
        Matrix partial_grad_W = blank_matrix(data_dim, n_classes, 0.);
        Matrix partial_grad_b = blank_matrix(1, n_classes, 0.);

        for (int epoch = 1; epoch <= n_epochs; ++epoch) {
            if (verbose && rank == 0)
                cout << string_format("Epoch %d / %d", epoch, n_epochs) << endl;

            for (int i = 0; i < n_rows(X); i+=batch_size) {
                // TODO: make random batch generator
                if (rank == 0) { // master
//                    cout << "master: will wait for workers for batch " << i << " to " << i+batch_size << endl;
                    // Clear the total gradient from previous communications
                    clear(total_grad_W);
                    clear(total_grad_b);

                    // Get partial gradients from each worker
                    // TODO don't wait for processes sequentially, take them in the order they are finished
                    for (int worker = 1; worker <= n_workers; ++worker) {
                        // Use the partial_grad  as a buffer in which to write matrix received from each worker sequentially

                        // Warning: relying on the fact that std::vector is stored in a contiguous block of memory
                        for (int row = 0; row < n_rows(partial_grad_W); ++row)
                            MPI_Recv(partial_grad_W[row].data(), n_cols(partial_grad_W), MPI_DOUBLE, worker, GRAD_W_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//                        MPI_Recv(&(partial_grad_W[0][0]), n_elements(partial_grad_W), MPI_DOUBLE, worker, GRAD_W_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Recv(partial_grad_b[0].data(), n_elements(partial_grad_b), MPI_DOUBLE, worker, GRAD_B_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//                        cout << "master: received from worker #" << worker << " partial grad W row = " << partial_grad_W[300] << endl;

                        // Add it to the sum
                        total_grad_W += partial_grad_W;
                        total_grad_b += partial_grad_b;

                    }

                    // After receiving all the partial gradients
//                    cout << "master: " << "total grad W row = " << total_grad_W[525] << "\t\ttotal grad b = " << total_grad_b;

                    W = W - lr * total_grad_W; // TODO regularization
                    b = b - lr * total_grad_b;
//                        cost_history.push_back(cost);

                    // Send back updated W and b to all workers
                    MPI_Bcast(b[0].data(), n_elements(b), MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
                    for (int row = 0; row < n_rows(partial_grad_W); ++row)
                        MPI_Bcast(W[row].data(), n_cols(W), MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
//                    cout << "master: send W row = " << W[300] << "\t\t b = " << b;


                }
                else { // workers
                    int start_index = i + (rank-1) * chunk_size;
                    int end_index = start_index + chunk_size;
//                    cout << "worker #" << rank << ": computing gradient on " << start_index<< " to " << end_index << endl;
                    Matrix chunk_X = chunk(X, start_index, end_index); // TODO: move this out of here
                    Matrix chunk_Y = chunk(Y, start_index, end_index);

                    if (n_rows(chunk_X) == 0) { // the number of samples is not divisible by the chunk size: remainder
                        // TODO: send status "nothing to do" instead of gradient zero and cost N/A
                        partial_grad_W = blank_matrix(data_dim, n_classes, 0.);
                        partial_grad_b = blank_matrix(1, n_classes, 0.);
                        cost = 0; // N/A
//                        cout << "worker #" << rank << ": no rows left" << endl;
                    }
                    else {
                        cost = grad(chunk_X, chunk_Y, partial_grad_W,  partial_grad_b);
//                        cout << "worker #" << rank << ": partial grad W row = " << partial_grad_W[300] << ",\t\tpartial grad b = " << partial_grad_b;
                    }

                    // Send partial gradients to master
                    for (int row = 0; row < n_rows(partial_grad_W); ++row)
                        MPI_Send(partial_grad_W[row].data(), n_cols(partial_grad_W), MPI_DOUBLE, MASTER, GRAD_W_TAG, MPI_COMM_WORLD);
//                    MPI_Send(&(partial_grad_W[0][0]), n_elements(partial_grad_W), MPI_DOUBLE, MASTER, GRAD_W_TAG, MPI_COMM_WORLD);
                    MPI_Send(partial_grad_b[0].data(), n_elements(partial_grad_b), MPI_DOUBLE, MASTER, GRAD_B_TAG, MPI_COMM_WORLD);
//                    cout << "worker #" << rank << ": sent partial b" << endl;


                    // Receive broadcasted W and b matrices
                    MPI_Bcast(b[0].data(), n_elements(b), MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
                    for (int row = 0; row < n_rows(partial_grad_W); ++row)
                        MPI_Bcast(W[row].data(), n_cols(W), MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
//                    cout << "worker #" << rank << ": received W row = " << W[300] << "\t\t b = " << b;
                }
//                break;


            }
            // TODO: add early stopping

            // Compute accuracy after each epoch
            acc = accuracy(predict(X), Y_labels);
            accuracy_history.push_back(acc);
        }
    }

    vector<int> predict(const Matrix& X) {
        Matrix Y_prob = softmax(X * W + b);
        return argmax_matrix(Y_prob);
    }

    int predict_one(const vector<double>& pixels, double& confidence) {
        // TODO: refactor to use .predict()
        Matrix X = {pixels};
        vector<double> y_prob = softmax(X * W + b)[0]; // just one image, take the first row

        int digit_predicted = argmax(y_prob);
        confidence = y_prob[digit_predicted];
        return digit_predicted;
    }


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