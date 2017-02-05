#include <iostream>
#include "reading.cpp"
#include "nn.cpp"

using namespace std;


int main() {
    Matrix images, labels;
    read_data("data/sample.csv", images, labels);
    print_image(images[0]);

//    NeuralNetwork net;
//    auto history = net.train(images, labels);
//    print(history);
}
