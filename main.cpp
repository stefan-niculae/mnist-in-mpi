#include <iostream>
#include "reading.cpp"
#include "nn.cpp"

using namespace std;

void train_and_save() {
    // TODO: names as additional parameters
    Matrix images, labels;
    read_data("data/train.csv", images, labels);

    NeuralNetwork net;
    vector<double> costs, accs;
    net.train(images, labels, costs, accs, 35, 100, .1);

    net.save("models/trained.nn");
    ofstream f("histories/training.txt");
    f << costs << endl << accs;
    f.close();
}

void predict(string pixels_string) {
    // TODO: names as additional parameters
//    NeuralNetwork net("models/trained_a.nn");
    vector<double> pixels = pixels_from_string(pixels_string);
    print_image(pixels);
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        cout << "usage: mnist_in_mpi.out (train|predict)" << endl;
        return 1;
    }

    if (string(argv[1]) == "train") {
        train_and_save();
        return 0;
    }

    if (string(argv[1]) == "predict") {
        if (argc < 3) {
            cout << "usage mnist_in_mpi.out predict <pixels>" << endl;
            return 1;
        }
        predict(argv[2]);
        return 0;
    }
}
