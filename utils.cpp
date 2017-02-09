#pragma once

#include <vector>
#include <string.h>
#include <stdarg.h>  // For va_start, etc.
#include <memory>    // For std::unique_ptr
#include <tuple>        // std::tuple, std::get, std::tie, std::ignore


using namespace std;

template <class T>
ostream& operator<<(ostream& os, const vector<T>& v) {
    os << '[';
    for (const auto& x : v)
        os << x << ' ';
    return os << ']';
}

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

void save_csv(tuple<vector<double>,vector<double>, vector<double>> histories, string path) {
    vector<double> acc_train, acc_test, costs;
    tie (acc_train, acc_test, costs) = histories;
    ofstream f(path);
    f << "epoch,accuracy_train,accuracy_test,cost" << endl; // header
    for (int i = 0; i < acc_train.size(); ++i)
        f << i+1 << "," << acc_train[i] << "," << acc_test[i] << "," << costs[i] << endl;
    f.close();
}

// for 2 (and 5 classes) ~> {0 0 1 0 0}
vector<double> make_one_hot(int value, int n_classes=10) {
    vector<double> result(n_classes, 0);
    result[value] = 1;
    return result;
}