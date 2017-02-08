#include <iostream>
#include <iomanip>
#include "mnist_io.cpp"
#include "nn.cpp"

using namespace std;


void train_and_save() {
    cout << setprecision(3);

    // Read
    int n_samples = 1000; Matrix images(n_samples, IMAGE_PIXELS), labels(n_samples, N_DIGITS);
    read_from_csv("data/test.csv", images, labels, false);    // TODO: names & params as additional args

    // Train
    NeuralNetwork net(IMAGE_PIXELS, N_DIGITS);
    auto histories = net.train(images, labels, 5, 200, 0.5);

    // Save
    net.save("models/trained.nn");
    save_csv(histories, "histories/training.csv");
}

void predict(string pixels_str) {
    // Parse
    Matrix input(1, IMAGE_PIXELS);  // only one image
    parse_image(pixels_str, input);

    // Read
    NeuralNetwork net("models/trained.nn");     // TODO: names as additional args

    // Predict
    double confidence;
//    int digit_predicted = net.predict_one(pixels, confidence);
//    cout << digit_predicted << endl << confidence << endl;
}

int main(int argc, const char* argv[]) {
    train_and_save();
    return 0;

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

