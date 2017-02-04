#include <iostream>
#include "utils.cpp"

using namespace std;


int main() {
    Matrix m = read_images("/Users/Stefan/Downloads/train_images", 10000, 28 * 28);
    print(m[0]);
    print(m[1]);
}
