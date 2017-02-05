#include <iostream>
#include "reading.cpp"
#include "nn.cpp"

using namespace std;


int main() {
    Matrix images, labels;
    read_data("/Users/Stefan/Downloads/sample.csv", images, labels);

    NeuralNetwork net;
    auto history = net.train(images, labels);

//    print(history);
}
