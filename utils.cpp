#pragma once

#include <vector>
#include <string.h>
#include <stdarg.h>  // For va_start, etc.
#include <memory>    // For std::unique_ptr

using namespace std;


// source: http://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
std::string string_format(const std::string fmt_str, ...) {
    int final_n, n = ((int)fmt_str.size()) * 2; /* Reserve two times as much as the length of the fmt_str */
    std::string str;
    std::unique_ptr<char[]> formatted;
    va_list ap;
    while(1) {
        formatted.reset(new char[n]); /* Wrap the plain char array into the unique_ptr */
        strcpy(&formatted[0], fmt_str.c_str());
        va_start(ap, fmt_str);
        final_n = vsnprintf(&formatted[0], n, fmt_str.c_str(), ap);
        va_end(ap);
        if (final_n < 0 || final_n >= n)
            n += abs(final_n - n + 1);
        else
            break;
    }
    return std::string(formatted.get());
}

template <class T>
ostream& operator<<(ostream& os, const vector<T>& v) {
    for (const auto& x : v)
        os << x << ' ';
    return os;
}

template <class T>
istream& operator>>(istream& is, vector<T>& v) {
    string line;
    getline(is, line);
    istringstream ss(line);

    T x;
    while (ss >> x) // until end of line
        v.push_back(x);

    return is;
}


// for 2 (and 5 classes) ~> {0 0 1 0 0}
vector<double> make_one_hot(int value, int n_classes=10) {
    vector<double> result(n_classes, 0);
    result[value] = 1;
    return result;
}

// {0 0 1 0 0} ~> 2
int from_one_hot(const vector<double>& v) {
    for (int i = 0; i < v.size(); ++i)
        if (v[i] == 1)
            return i;

    throw runtime_error("Vector to convert from one-hot form contains no ones");
}


int argmax(vector<double> v) {
    double max = v[0];
    int max_idx = 0;

    for (int i = 1; i < v.size(); ++i)
        if (v[i] > max) {
            max = v[i];
            max_idx = i;
        }

    return max_idx;
}
