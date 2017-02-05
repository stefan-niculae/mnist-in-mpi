#pragma once

#include <vector>
#include "utils.cpp"

using namespace std;

typedef vector<double> Vector;


template <class T>
void print(const vector<T>& v) {
    for (const auto& x : v)
        cout << x << ' ';
    cout << endl;
}

void print_image(const Vector& pixels) {
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

// scalar-vector multiplication
template <class T>
vector<T> operator* (T scalar, const vector<T>& vector) {
    auto result = vector;
    for (auto& elem : result)
        elem *= scalar;
    return result;
}

// vector subtraction
template <class T>
vector<T> operator- (const vector<T>& lhs, const <vector>& rhs) {
    if (lhs.size() != rhs.size())
        throw runtime_error(string_format("Vector subtraction: sizes are different: "
                                                  "lhs = %d, rhs = %d", lhs.size(), rhs.size()));

    vector<T> result = lhs;
    for (const auto& x : rhs)
        result -= x;
    return result;
}