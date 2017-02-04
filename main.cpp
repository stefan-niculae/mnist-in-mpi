#include <iostream>
#include <vector>
#include "utils.cpp"

using namespace std;


int main() {
    vector<int> v0 = {0, 2, 3};
    vector<int> v1 = {1, 5, 6};
    vector<int> v2 = {2, 0, 1};

    vector<vector<int>> m = {v0, v1, v2};

    print(m * m);
}
