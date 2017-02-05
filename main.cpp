#include <iostream>
#include "reading.cpp"
#include "nn.cpp"

using namespace std;


int main() {
    Matrix images, labels;
    read_data("data/train.csv", images, labels);

    NeuralNetwork net;
    vector<double> costs, accs;
    net.train(images, labels, costs, accs, 100, 80, .075);

    net.save("models/trained.nn");
    ofstream f("histories/training.txt");
    f << costs << endl << accs;
    f.close();
}
