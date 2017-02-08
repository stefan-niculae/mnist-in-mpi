#include <iostream>
#include "reading.cpp"
#include "nn.cpp"
#include "matrix.cpp"
#include <iomanip>

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

    cout << setprecision(3);

    int n_samples = 2000; Matrix images(n_samples, 784), labels(n_samples, 10);
    read_from_csv("data/test.csv", images, labels, false);
    NeuralNetwork net(784, 10);
    auto histories = net.train(images, labels, 100, 200, 0.5, true, true, true);
}

