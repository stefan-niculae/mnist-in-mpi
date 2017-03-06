#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "matrix.cpp"
#include "utils.cpp"

using namespace std;

const int PIXEL_DIM = 255;

void read_data(string filename, Matrix& images, Matrix& labels, bool verbose= true) {
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
    file.close();

    for (auto val : label_values)
        labels.push_back(make_one_hot(val));

    if (verbose)
        cout << "Done reading from " + filename << endl;
}

void print_image(const vector<double>& pixels) {
    for (int i = 0; i < 28 * 28; ++i) {
        if (i > 0 && i % 28 == 0)
            cout << endl << endl;

        if (pixels[i] == 0)
            cout << "  ";
        else
            cout << int(pixels[i] * 100); // turns 0.25 into 25
        cout << ' ';
    }
    cout << endl << endl;
}
