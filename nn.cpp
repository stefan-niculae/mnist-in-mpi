#include "utils.cpp"
#include <vector>
#include <utility>
using namespace std;


class NN {

    Matrix W;
    Vector b;

    public:

        pair<Matrix, Vector> grad(Matrix X, Matrix Y) {
            Matrix Y_prob = softmax(W * X + b));
            // error at last layer
            double cost = CE(Y, Y_prob);
            Matrix delta = Y_prob - Y;
            // return gradient of cros entropy cost
            return make_pair(delta * transpose(X), delta);
        }

        void train(Matrix X, Matrix Y, int epochs=10, int batch_size=100, double lr=0.1) {
            for(int epoch=0; epoch<epochs; epoch++) {
                for(int i=0; i<n_rows(X); i+=batch_size) {
                    
                    pair<Matrix, Vector> deltaJ;
                    deltaJ = grad(chunk(X, i, i+batch_size), chunk(Y, i, i+batch_size);
                    W = W - lr * 1/batch_size * deltaJ.first;
                    b = b - lr * 1/batch_size * deltaJ.second;
                }
            }
        }
};
