#pragma once

#include <vector>

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

template <class T>
vector<T> operator* (T scalar, const vector<T>& vector) {
    auto result = vector;
    for (auto& elem : result)
        elem *= scalar;
    return result;
}
