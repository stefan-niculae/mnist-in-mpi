#pragma once

#include <random> // normal_distribution
#include <iostream>
#include <fstream>
#include "matrix.cpp"
#include "evaluate.cpp"
#include <vector>
#include <chrono> // stopwatch
#include "mpi.h"
#include <utility> // pair


using namespace std;


/*** MPI constants ***/
const int MASTER = 0;
const int GRAD_W_TAG = 0;
const int GRAD_B_TAG = 1;
const int COST_TAG   = 2;

static vector<double> NONE = vector<double>();



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

double cross_entropy(const Matrix& Y, const Matrix& Y_prob) {
    // Cost function
    Matrix logY_prob(Y.n_rows, Y.n_cols);
    Matrix result(Y.n_rows, Y.n_cols);
    log(Y_prob, logY_prob);
    hadamard(Y, logY_prob, result);
    return -1/double(Y.n_rows) * sum(result);
}

void random_init(Matrix& W, double mean=0., double std=1.) {
    default_random_engine generator;
    normal_distribution<double> distribution(mean, std);

    for (int i = 0; i < W.n_rows; ++i)
        for (int j = 0; j < W.n_cols; ++j)
            W.data[i][j] = distribution(generator);
}




class NeuralNetwork {

private:

public:
    Matrix b;
    Matrix W;        // TODO? more layers

    const int N_CLASSES;
    const int DATA_DIM;

    NeuralNetwork(const int data_dim, const int n_classes) :
            N_CLASSES(n_classes), DATA_DIM (data_dim),
            W(data_dim, n_classes), b(1, n_classes) { }

    NeuralNetwork(string path) :
        N_CLASSES(10), DATA_DIM(784),
        W(784, 10), b(1, 10) // TODO
    {
        this->load(path);
    }

    double grad(const Matrix& chunk_X, const Matrix& chunk_Y,
                Matrix& partial_grad_W, Matrix& partial_grad_b,
                Matrix& cXW, Matrix& cXWb, Matrix& cY_prob, Matrix& delta, Matrix& trX, Matrix& trXdelta, Matrix& delta_sums,
                const bool compute_cost) {
        dot(chunk_X, W, cXW); // cXW = X * W
        add_to_each(cXW, b, cXWb); // cXWb = X * W + b
        softmax(cXWb, cY_prob); // cY_prob = softmax(X * W + b)
        sub(cY_prob, chunk_Y, delta); // delta = cY_prob - Y

        double contribution = 1/double(chunk_X.n_rows);

        transpose(chunk_X, trX); // trX = transpose(X)
        dot(trX, delta, trXdelta); // trXdelta = transpose(X) * delta
        scalar_mult(contribution, trXdelta, partial_grad_W); // grad_W = 1/n * transpose(chunk_X) * delta

        col_wise_sums(delta, delta_sums);
        scalar_mult(contribution, delta_sums, partial_grad_b); // grad_b = 1/n * col_wise_sums(delta)

        return compute_cost ? cross_entropy(chunk_Y, cY_prob) : 0;
    }

    pair<vector<double>, vector<double>>
    train(const Matrix& X, const Matrix& Y,
          const int n_epochs=100, const int batch_size=200, const double lr=0.1,
          bool compute_acc=true, bool compute_cost=true, bool verbose=true) {

        const int n_samples = X.n_rows;
        const int data_dim = X.n_cols;
        const int n_classes = Y.n_cols;

        // MPI
        int rank, n_processes;
        MPI_Init(NULL, NULL);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_processes);
        const int n_workers = n_processes - 1; // how many processes compute gradient in parallel, first is master
        if (n_workers < 1) throw runtime_error("no workers!");

        const int chunk_size = batch_size / n_workers; // how many rows each worker gets
        if (rank != MASTER) verbose = false; // only master is allowed to talk

        if (n_samples % chunk_size != 0) throw runtime_error("samples not divisible by chunk");

        // Intermediary matrices used by master
        Matrix lr_grad_W(W.n_rows, W.n_cols),       lr_grad_b(b.n_rows, b.n_cols);
        Matrix partial_grad_W(W.n_rows, W.n_cols),  partial_grad_b(b.n_rows, b.n_cols);
        Matrix total_grad_W(W.n_rows, W.n_cols),    total_grad_b(b.n_rows, b.n_cols);

        // Intermediary matrices used reused for every grad call by workers
        Matrix chunk_X(chunk_size, data_dim), chunk_Y(chunk_size, n_classes);
        // chunk_X * W ie (CHUNK_SIZE, DATA_DIM) * (DATA_DIM, N_CLASSES)
        Matrix cXW(chunk_size, n_classes), cXWb(chunk_size, n_classes), cY_prob(chunk_size, n_classes), delta(chunk_size, n_classes);
        Matrix trX(data_dim, chunk_size); // transpose CHUNK_SIZE, DATA_DIM
        Matrix trXdelta(data_dim, n_classes); // DATA_DIM, CHUNK_SIZE * CHUNK_SIZE, N_CLASSES
        Matrix delta_sums(b.n_rows, b.n_cols);

        chrono::time_point<chrono::system_clock> start_time, end_time;
        double acc, total_cost, partial_cost;
        vector<double> accuracy_history, cost_history;
        vector<int> Y_labels = labels_from_one_hot(Y); // {5, 2, 9, ... }
        int start_index;

        // Initial weights
        random_init(W);

        if (verbose)
            cout << "Training on " << n_samples / 1000 << "k samples"
                 << " with " << n_workers << " workers"
                 << " (" << chunk_size << " chunk size)" << endl;

        for (int epoch = 1; epoch <= n_epochs; ++epoch) {
            if (verbose) {
                cout << string_format("Epoch %d/%d", epoch, n_epochs) << flush;
                start_time = chrono::system_clock::now();
            }

            for (int batch_start = 0; batch_start < X.n_rows; batch_start += batch_size) {
                if (rank == MASTER) {

                    // clear summed from previous batch
                    total_grad_W.clear();
                    total_grad_b.clear();
                    if (compute_cost) total_cost = 0;

                    // Get partial gradients from each worker
                    for (int worker = 1; worker <= n_workers; ++worker) {
                        MPI_Recv(partial_grad_W.data[0], partial_grad_W.n_elements, MPI_DOUBLE, worker, GRAD_W_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Recv(partial_grad_b.data[0], partial_grad_b.n_elements, MPI_DOUBLE, worker, GRAD_B_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        if (compute_cost) MPI_Recv(&partial_cost, 1, MPI_DOUBLE, worker, COST_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                        // Sum partial gradients
                        add_to(total_grad_W, partial_grad_W);
                        add_to(total_grad_b, partial_grad_b);
                        if (compute_cost) total_cost += partial_cost;
                    }

                    // After receive
                    scalar_mult(lr, total_grad_W, lr_grad_W);                     // TODO? regularization
                    scalar_mult(lr, total_grad_b, lr_grad_b);
                    sub_from(W, lr_grad_W); // W = W - lr * grad_W
                    sub_from(b, lr_grad_b); // b = b - lr * grad_b

                    if (compute_cost) cost_history.push_back(total_cost);
                }


                if (rank != MASTER) { // worker
                    start_index = batch_start + (rank - 1) * chunk_size;                    // TODO? make random batch generator
                    take_chunk(X, start_index, chunk_X); // chunk_X = X[batch_start ... batch_start + CHUNK_SIZE]
                    take_chunk(Y, start_index, chunk_Y);
                    partial_cost = grad(chunk_X, chunk_Y,
                                        partial_grad_W, partial_grad_b,
                                        cXW, cXWb, cY_prob, delta, trX, trXdelta, delta_sums,
                                        compute_cost); // grad_W and grad_b are now filled with result
                    // Send partial gradients to master
                    MPI_Send(partial_grad_W.data[0], partial_grad_W.n_elements, MPI_DOUBLE, MASTER, GRAD_W_TAG, MPI_COMM_WORLD);
                    MPI_Send(partial_grad_b.data[0], partial_grad_b.n_elements, MPI_DOUBLE, MASTER, GRAD_B_TAG, MPI_COMM_WORLD);
                    if (compute_cost) MPI_Send(&partial_cost, 1, MPI_DOUBLE, MASTER, COST_TAG, MPI_COMM_WORLD);
                }

                // Send back updated W and b to all workers
                MPI_Bcast(W.data[0], W.n_elements, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
                MPI_Bcast(b.data[0], b.n_elements, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
            }

            if (rank == MASTER && compute_acc) {
                // TODO: add early stopping
                acc = accuracy(predict(X), Y_labels);
                accuracy_history.push_back(acc);
            }

            if (verbose) {
                end_time = chrono::system_clock::now();
                chrono::duration<double> elapsed_seconds = end_time - start_time;
                cout << "\t\tdone in " << elapsed_seconds.count() << "s" << flush;
                if (compute_acc)  cout << "\t\taccuracy: " << acc * 100 << "%" << flush;
                if (compute_cost) cout << "\t\tcost: " << total_cost;
                cout << endl;
            }
        }

        MPI_Finalize();

        // TODO: print total time and average epoch time, max accuracy
        return make_pair(accuracy_history, cost_history);
    }

    vector<int> predict(const Matrix& X) {
        const int n_samples = X.n_rows, n_classes = b.n_cols;
        Matrix XW = Matrix(n_samples, n_classes), XWb(n_samples, n_classes), Y_prob(n_samples, n_classes);

        dot(X, W, XW); // XW = X * W
        add_to_each(XW, b, XWb); // XWb = X * W + b
        softmax(XWb, Y_prob); // Y_prob = softmax(X * W + b)
        return argmax(Y_prob);
    }

    pair<int, double> predict_one(const Matrix& X) {
        // TODO: refactor to use NeuralNetwork::predict
        const int n_samples = X.n_rows, n_classes = b.n_cols;
        Matrix XW = Matrix(n_samples, n_classes), XWb(n_samples, n_classes), Y_prob(n_samples, n_classes);

        cout << X;
        cout << W;
        cout << XW;

        dot(X, W, XW); // XW = X * W
        add_to_each(XW, b, XWb); // XWb = X * W + b
        softmax(XWb, Y_prob); // Y_prob = softmax(X * W + b)

        int digit_predicted = argmax(Y_prob)[0]; // first image, the only one
        double confidence = Y_prob.data[0][digit_predicted];
        return make_pair(digit_predicted, confidence);
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
    os << net.N_CLASSES << ' ' << net.DATA_DIM << endl;
    os << net.b << net.W;  // place W last so it can be read until end of file
    return os;
}

istream& operator>>(istream& is, NeuralNetwork& net) {
    // net.b is a 1xn matrix, if we read a matrix normally, we read until end of file,
    // but a vector is read until end of file
    int n_classses, data_dim;
    is >> n_classses >> data_dim;
    is >> net.b >> net.W;
    return is;
}