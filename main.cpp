#include <iostream>
#include "reading.cpp"
#include "nn.cpp"
#include "matrix.cpp"

using namespace std;

//void train_and_save() {
//    // TODO: names as additional parameters
//    Matrix images, labels;
//    read_data("data/train.csv", images, labels);
//
//    NeuralNetwork net;
//    vector<double> costs, accs;
//    net.train(images, labels, costs, accs, 35, 100, .1);
//
//    net.save("models/trained.nn");
//    ofstream f("histories/training.txt");
//    f << costs << endl << accs;
//    f.close();
//}

//int predict(string pixels_string) {
//    // TODO: names as additional parameters
//    vector<double> pixels = pixels_from_string(pixels_string);
//    NeuralNetwork net("models/trained_a.nn");
//
//    double confidence;
//    int digit_predicted = net.predict_one(pixels, confidence);
//    cout << digit_predicted << endl << confidence << endl;
//
//    return digit_predicted;
//}

int main(int argc, const char* argv[]) {
//    if (argc < 2) {
//        cout << "usage: mnist_in_mpi.out (train|predict)" << endl;
//        return 1;
//    }
//
//    if (string(argv[1]) == "train") {
//        train_and_save();
//        return 0;
//    }
//
//    if (string(argv[1]) == "predict") {
//        if (argc < 3) {
//            cout << "usage mnist_in_mpi.out predict <pixels>" << endl;
//            return 1;
//        }
//        predict(argv[2]);
//        return 0;
//    }

//    double* v = new double[5];
//    for (int i = 0; i < 5; ++i)
//        v[i] = 100+i;
//    double* w = v+2;
//    cout << w[0];

    int n_samples = 60000;
    Matrix images(n_samples, 784), labels(n_samples, 10);
    read_from_csv(string("/home/ionut/workspace/ppc/mnist-in-mpi/data/sample.csv"), images, labels);
    NeuralNetwork net;
    vector<double> cost_history, acc_history;
    net.train(images, labels, cost_history,acc_history);
    for (int i = 0; i < acc_history.size(); ++i) {
        cout << acc_history[i] << endl;

    }


//    Matrix m(4, 2);
//    int k = 0;
//    for (int i = 0; i < m.n_rows; ++i)
//        for (int j = 0; j < m.n_cols; ++j)
//            m.data[i][j] = k++;
//    cout << m;
//
//    Matrix chunk(2, 2);
//    take_chunk(m, 1, chunk);
//    cout << endl << chunk;


    return 0;
}
