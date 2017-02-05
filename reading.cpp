#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include "matrix.cpp"
#include "utils.cpp"

using namespace std;

const int PIXEL_DIM = 255;

vector<double> make_one_hot(int value, int n_classes=10) {
    vector<double> result(n_classes, 0);
    result[value] = 1;
    return result;
}

void read_data(string filename, Matrix& images, Matrix& labels) {
    ifstream file(filename);

    if (!file.is_open())
        throw runtime_error(string_format("Read MNIST data: could not open file: " + filename));

    vector<int> label_values;
    string line;
    while (getline(file, line)) {
        istringstream line_stream(line);

        int label;
        line_stream >> label;
        label_values.push_back(label);
        line_stream.ignore(); // skip comma after label

        vector<double> image;
        int pixel;
        while (line_stream >> pixel) {
            image.push_back(double(pixel) / PIXEL_DIM);
            line_stream.ignore(); // skip comma after pixel value
        }

        images.push_back(image);
    }

    for (auto val : label_values)
        labels.push_back(make_one_hot(val));
}
