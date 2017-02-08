#pragma once

#include <random> // normal_distribution
#include <iostream>
#include <fstream>
#include "matrix.cpp"
#include "evaluate.cpp"
#include <vector>
#include <chrono>
#include "mpi.h"


using namespace std;


/*** MPI constants ***/
const int MASTER = 0;
const int GRAD_W_TAG    = 0;
const int GRAD_B_TAG    = 1;
const int COST_TAG      = 2;



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

    const int N_SAMPLES = 10000;
    const int N_WORKERS = 5; // how many processes compute gradient in parallel

    const int N_CLASSES = 10; // 10 digits from 0 to 9
    const int DATA_DIM = 784; // 28*28 = 784 pixels for an image
    const int BATCH_SIZE = 500; // after how many pictures the weights are updated
    const int CHUNK_SIZE = BATCH_SIZE / N_WORKERS; // how many rows each worker gets

    // X dimensions: N_SAMPLES x DATA_DIM
    // Y dimensions: N_SAMPLES x N_CLASSES

    Matrix W = Matrix(DATA_DIM, N_CLASSES);        // TODO? more layers
    Matrix b = Matrix(1, N_CLASSES);

    Matrix chunk_X = Matrix(CHUNK_SIZE, DATA_DIM);
    Matrix chunk_Y = Matrix(CHUNK_SIZE, N_CLASSES);
    // chunk sized
        Matrix cXW = Matrix(CHUNK_SIZE, N_CLASSES); // chunk_X * W ie (CHUNK_SIZE, DATA_DIM) * (DATA_DIM, N_CLASSES)
        Matrix cXWb = Matrix(CHUNK_SIZE, N_CLASSES);
        Matrix cY_prob = Matrix(CHUNK_SIZE, N_CLASSES);
        Matrix delta = Matrix(CHUNK_SIZE, N_CLASSES);

        Matrix trX = Matrix(DATA_DIM, CHUNK_SIZE); // transpose CHUNK_SIZE, DATA_DIM
        Matrix trXdelta = Matrix(DATA_DIM, N_CLASSES); // DATA_DIM, CHUNK_SIZE * CHUNK_SIZE, N_CLASSES

        Matrix delta_sums = Matrix(1, N_CLASSES); // like b


    Matrix lr_grad_W = Matrix(DATA_DIM, N_CLASSES); // like W
    Matrix lr_grad_b = Matrix(1, N_CLASSES); // like b

    Matrix partial_grad_W = Matrix(DATA_DIM, N_CLASSES); // like W
    Matrix partial_grad_b = Matrix(1, N_CLASSES); // like b

    Matrix total_grad_W = Matrix(DATA_DIM, N_CLASSES); // like W
    Matrix total_grad_b = Matrix(1, N_CLASSES); // like b

    // for prediction, on entire dataset
    Matrix XW = Matrix(N_SAMPLES, N_CLASSES);
    Matrix XWb = Matrix(N_SAMPLES, N_CLASSES);
    Matrix Y_prob = Matrix(N_SAMPLES, N_CLASSES);


public:

    NeuralNetwork() { }

//    NeuralNetwork(string path) {
//        this->load(path);
//    }

    double grad() {
        dot(chunk_X, W, cXW); // cXW = X * W
        add_to_each(cXW, b, cXWb); // cXWb = X * W + b
        softmax(cXWb, cY_prob); // cY_prob = softmax(X * W + b)
        sub(cY_prob, chunk_Y, delta); // delta = cY_prob - Y

        double contribution = 1/double(CHUNK_SIZE);

        transpose(chunk_X, trX); // trX = transpose(X)
        dot(trX, delta, trXdelta); // trXdelta = transpose(X) * delta
        scalar_mult(contribution, trXdelta, partial_grad_W); // grad_W = 1/n * transpose(chunk_X) * delta

        col_wise_sums(delta, delta_sums);
        scalar_mult(contribution, delta_sums, partial_grad_b); // grad_b = 1/n * col_wise_sums(delta)

        // TODO maybe
        return cross_entropy(chunk_Y, cY_prob); // cost
    }

    void train(const Matrix& X, const Matrix& Y,
               vector<double>& cost_history, vector<double>& accuracy_history,
               int n_epochs=100, double lr=0.1,
               bool verbose=true) {
        if (N_SAMPLES != X.n_rows) throw runtime_error("bad N_SAMPLES");

        double total_cost=0, local_cost=0, acc;
        vector<int> Y_labels = labels_from_one_hot(Y); // {5, 2, 9, ... }
        std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
        MPI_Init(NULL, NULL);
        int rank, n_processes;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_processes);
        if (n_processes != N_WORKERS + 1) throw runtime_error("bad N_WORKERS");
        if (N_SAMPLES % CHUNK_SIZE != 0) throw runtime_error("samples not divisible by chunk");
        int start_index;

        random_init(W);

        for (int epoch = 1; epoch <= n_epochs; ++epoch) {
            if (rank == MASTER) {
                cout << string_format("Epoch %d / %d", epoch, n_epochs) << flush;
                start_time = std::chrono::system_clock::now();
            }

            for (int batch_start = 0; batch_start < X.n_rows; batch_start += BATCH_SIZE) {
                if (rank == MASTER) {

                    // clear summed from previous batch
                    total_grad_W.clear();
                    total_grad_b.clear();
                    total_cost = 0;

                    // Get partial gradients from each worker
                    for (int worker = 1; worker <= N_WORKERS; ++worker) {
//                        cout << "master: waiting for worker #" << worker << " in epoch " << epoch << endl;
                        MPI_Recv(partial_grad_W.data[0], partial_grad_W.n_elements, MPI_DOUBLE,
                                 worker, GRAD_W_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Recv(partial_grad_b.data[0], partial_grad_b.n_elements, MPI_DOUBLE,
                                 worker, GRAD_B_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Recv(&local_cost, 1, MPI_DOUBLE,
                                 worker, COST_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        // Sum partial gradients
                        add_to(total_grad_W, partial_grad_W);
                        add_to(total_grad_b, partial_grad_b);
                        total_cost += local_cost;
                    }

                    // After receive
                    // TODO? regularization
                    scalar_mult(lr, total_grad_W, lr_grad_W); // TODO!!! minus here!?
                    scalar_mult(lr, total_grad_b, lr_grad_b);
                    sub_from(W, lr_grad_W); // W = W - lr * grad_W
                    sub_from(b, lr_grad_b); // b = b - lr * grad_b

                    cost_history.push_back(total_cost); // TODO
                }


                if (rank != MASTER) { // worker
                    // TODO? make random batch generator
                    start_index = batch_start + (rank - 1) * CHUNK_SIZE;
//                    cout << "worker #" << rank << ": batch_start " << batch_start << ", start_index = " << start_index << endl;
                    take_chunk(X, start_index, chunk_X); // chunk_X = X[batch_start ... batch_start + CHUNK_SIZE]
                    take_chunk(Y, start_index, chunk_Y);
                    local_cost = grad(); // grad_W and grad_b are now filled with result
                    // Send partial gradients to master
                    MPI_Send(partial_grad_W.data[0], partial_grad_W.n_elements, MPI_DOUBLE,
                             MASTER, GRAD_W_TAG, MPI_COMM_WORLD);
                    MPI_Send(partial_grad_b.data[0], partial_grad_b.n_elements, MPI_DOUBLE,
                            MASTER, GRAD_B_TAG, MPI_COMM_WORLD);
                    MPI_Send(&local_cost, 1, MPI_DOUBLE,
                             MASTER, COST_TAG, MPI_COMM_WORLD);
                }

                // Send back updated W and b to all workers
                MPI_Bcast(W.data[0], W.n_elements, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
                MPI_Bcast(b.data[0], b.n_elements, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

            }

            if (rank == MASTER) {
                end_time = std::chrono::system_clock::now();
                std::chrono::duration<double> elapsed_seconds = end_time - start_time;
                cout << " done in " << elapsed_seconds.count() << "s, " << flush;
            }

            if (rank == MASTER) {
                // TODO: add early stopping
                // Compute accuracy after each epoch
                acc = accuracy(predict(X), Y_labels);
                accuracy_history.push_back(acc);
                cout << "accuracy: " << acc * 100 << "%" <<", cost:" << total_cost << endl;
            }
        }

        MPI_Finalize();
    }

    vector<int> predict(const Matrix& X) {
        dot(X, W, XW); // XW = X * W
        add_to_each(XW, b, XWb); // XWb = X * W + b
        softmax(XWb, Y_prob); // Y_prob = softmax(X * W + b)
        return argmax(Y_prob);
    }
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