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
    auto prediction = net.predict_one(input);
    cout << prediction.first << endl << prediction.second << endl;
}

int main(int argc, const char* argv[]) {
//    string dummy = "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.035681610247026534,0.14468696902365705,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.7847340216965103,0.9996078943928898,0.3182590511044308,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.9332113449222323,0.9998692981309633,0.46660567246111617,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.9332113449222323,0.9998692981309633,0.46660567246111617,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.9332113449222323,0.9998692981309633,0.46660567246111617,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.9465429355639786,0.9998692981309633,0.4667363743301529,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.10351588027708797,0.9998692981309633,0.9998692981309633,0.3609985622794406,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.4506600444386355,0.9998692981309633,0.9998692981309633,0.1388053849170043,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.8065612338256437,0.9998692981309633,0.8070840413017906,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.18664226898444647,1,0.9998692981309633,0.4289635341785388,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.508822376159979,0.9998692981309633,0.9828780551561888,0.07240883544634688,0,0,0,0,0,0.07515357469611815,0.7183374722258529,0.5804470003921056,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.730231342308195,0.9998692981309633,0.760423474055679,0,0,0,0,0,0,0.32910730623447915,0.9998692981309633,0.9998692981309633,0.06247549339955561,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.008626323356423996,0.9170043131616782,0.9998692981309633,0.6095935171872958,0.06665795320873089,0.04012547379427526,0,0,0,0,0.3332897660436544,0.9998692981309633,0.9998692981309633,0.06665795320873089,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.23683178669454974,0.9998692981309633,0.9998692981309633,0.9998692981309633,0.9998692981309633,1,0.9244543196967716,0.7796366488040779,0.6227944059600052,0.4624232126519409,0.5279048490393413,0.9998692981309633,0.9998692981309633,0.06665795320873089,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.38726963795582275,0.9998692981309633,0.9998692981309633,0.9998692981309633,0.9998692981309633,0.9998692981309633,0.9998692981309633,0.9998692981309633,0.9998692981309633,0.9998692981309633,0.9998692981309633,0.9998692981309633,0.9998692981309633,0.7551953992942099,0.5749575218925631,0.4263494967978042,0.24924846425303882,0.20088877270944974,0.04025617566331199,0,0,0,0,0,0,0,0,0,0.06116847470918834,0.6073715854136714,0.47405567899620965,0.3332897660436544,0.3332897660436544,0.39158279963403475,0.5177101032544765,0.6884067442164423,0.8287805515618873,0.9773885766566461,0.9998692981309633,0.9998692981309633,0.9998692981309633,0.9998692981309633,0.9998692981309633,0.9998692981309633,0.9998692981309633,0.9998692981309633,0.9030192131747484,0.06822637563717161,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.03764213828257744,0.4988890341131878,0.9998692981309633,0.9998692981309633,0.7297085348320481,0.8572735590118938,0.9909815710364658,0.9998692981309633,0.9998692981309633,0.9878447261795844,0.12155273820415632,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.3332897660436544,0.9998692981309633,0.9998692981309633,0.06665795320873089,0,0.058162331721343614,0.18232910730623447,0.24950986799111227,0.19095543066265847,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.3332897660436544,0.9998692981309633,0.9998692981309633,0.06665795320873089,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.3332897660436544,0.9998692981309633,0.9998692981309633,0.06665795320873089,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.3332897660436544,0.9998692981309633,0.9998692981309633,0.06665795320873089,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.3332897660436544,0.9998692981309633,0.9998692981309633,0.06665795320873089,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.3332897660436544,0.9998692981309633,0.9998692981309633,0.06665795320873089,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.3332897660436544,0.9998692981309633,0.9998692981309633,0.06665795320873089,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.3332897660436544,0.9998692981309633,0.9998692981309633,0.06665795320873089,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.15422820546333812,0.9417069664096197,0.8145340478368841,0.01463860933211345,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.027970199973859626,0.009933342046791268,0,0,0,0,0,0,0,0,0";
//    predict(dummy);

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

