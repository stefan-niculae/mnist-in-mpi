#include <iostream>
#include "utils.cpp"

using namespace std;


int main() {
    Matrix m = read_images("/Users/Stefan/Downloads/t10k-images-idx3-ubyte", 10, 28 * 28);
    print_image(m[8]);
    cout << m[8].size();
}
