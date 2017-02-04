#include <vector>
#include <algorithm>  // max_element
#include <math.h>  // exp
#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>


using namespace std;

typedef vector<double> Vector;
typedef vector<vector<double>> Matrix;


template <class T>
vector<T> softmax(vector<T> v) {
    // Subtract maximum to avoid overflow
    double max = *max_element(v.begin(), v.end());

    auto expd = v; // copy, does not mutate v
    double sum = 0;
    for (double &x : expd) {
        x = exp(x - max);
        sum += x;
    }

    vector<T> result;
    for (double x : expd) {
        result.push_back(x / sum);
    }

    return result;
}

template <class T>
vector<vector<T>> softmax(vector<vector<T>> m) {
    auto result = m;
    for (auto& row : m)
        row = softmax(row);
}



template <class T>
vector<T> chunk(vector<T> m, int from, int to) {
    return vector<T>(&m[from], &m[to + 1]);;
}

template <class T>
int n_rows(vector<vector<T>> m) {
    return m.size();
}

template <class T>
int n_cols(vector<vector<T>> m) {
    return n_rows(m) == 0 ?
        0 :
        m[0].size();
}

template <class T>
vector<vector<T>> blank_matrix(int rows, int cols, T value) {
    return vector<vector<T>>(rows, vector<T>(cols, value));
}

template <class T>
vector<vector<T>> operator* (double scalar, vector<vector<T>> matrix) {
    auto result = matrix;
    for (auto &row : result)
        for (auto& elem : row)
            elem *= scalar;
    return result;
}

template <class T>
vector<vector<T>> operator- (vector<vector<T>> lhs, vector<vector<T>> rhs) {
    if(n_rows(lhs) != n_rows(rhs)) {
        throw sprintf("Number of rows is different, %d != %d", n_rows(lhs), n_rows(rhs));
    } else if(n_cols(lhs) != n_cols(rhs)) {
        throw sprintf("Number of colums is different, %d != %d", n_cols(lhs), n_cols(rhs));
    }
    auto result = lhs;
    for (int i = 0; i < n_rows(rhs); ++i)
        for (int j = 0; j < n_cols(rhs); ++j)
            result[i][j] -= rhs[i][j];
    return result;
}

template <class T>
vector<vector<T>> operator* (vector<vector<T>> lhs, vector<vector<T>> rhs) {
    if (n_cols(lhs) != n_rows(rhs))
        throw "Dimensions do not agree for matrix multiplication!";

    int n = n_rows(lhs);
    int p = n_cols(lhs);
    int m = n_cols(rhs);
    auto result = blank_matrix(n, m, 0);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < p; ++k)
                result[i][j] += lhs[i][k] * rhs[k][j];

    return result;
}




template <class T>
void print(vector<T> v) {
    for (auto x : v)
        cout << x << ' ';
    cout << endl;
}

template <class T>
void print(vector<vector<T>> m) {
    for (auto row : m)
        print(row);
    cout << endl;
}

void print_image(Vector pixels) {
    for (int i = 0; i < 28 * 28; ++i) {
        if (i > 0 && i % 28 == 0)
            cout << endl;

        if (pixels[i] == 0)
            cout << ' ';
        else
            cout << pixels[i];
        cout << ' ';
    }
    cout << endl << endl;
}

// source: https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set/
int reverse_int(int x) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 =  x      & 255;
    ch2 = (x>>8)  & 255;
    ch3 = (x>>16) & 255;
    ch4 = (x>>24) & 255;
    return ((int)ch1 << 24) +
           ((int)ch2 << 16) +
           ((int)ch3 << 8)  +
           ch4;
}

Matrix read_images(string path, int n_images, int image_size) {
    Matrix result = blank_matrix(n_images, image_size, 0.);
    ifstream file(path, ios::binary);

    if (!file.is_open())
        throw "Incorrect path: " + path;

    int magic_number, n_rows, n_cols;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);

    file.read((char*)&n_images, sizeof(n_images));
    n_images = reverse_int(n_images);

    file.read((char*)&n_rows, sizeof(n_rows));
    n_rows = reverse_int(n_rows);

    file.read((char*)&n_cols, sizeof(n_cols));
    n_cols = reverse_int(n_cols);

    for (int i = 0; i < n_images; ++i)
        for (int r = 0; r < n_rows; ++r)
            for (int c = 0; c < n_cols; ++c) {
                unsigned char temp;
                file.read((char*)&temp, sizeof(temp));
                result[i][(n_rows*r)+c] = (double)temp;
            }

    return result;
}
