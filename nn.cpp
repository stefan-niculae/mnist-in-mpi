#include "matrix.cpp"
#include <vector>
using namespace std;

class NN {

    vector<vector<double> > W;
    vector<double> b;

    public:

        Matrix grad(Matrix X, vector<bool> y) {
            Matrix y_prob = softmax(add(dot(W, X), b));
            // error at last layer
            Matrix delta = sub(y_prob, y);
            // return gradient of cros entropy cost
            return dot(delta, transpose(X));
        }
        void train(vector<vector<double> > X, vector<vector<bool> > Y, int epochs=10, int batch_size=100, double lr=0.1) {
            for(int epoch=0;epoch<epochs;epoch++) {
                for(int i=0;i<n_rows(X);i+=batch_size) {
                    
                    deltaJ = grad(chunk(X, i, i+batch_size), chunk(Y, i, i+batch_size);
                    W = W - lr * deltaJ;
                }
            }
        }

};
