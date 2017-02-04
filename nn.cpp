#include "utils.cpp"
#include <vector>
#include <utility>
using namespace std;

class NN {

    vector<vector<double> > W;
    vector<double> b;

    public:

        pair<vector<vector<double> >, vector<double> > grad(vector<vector<double> > X, vector<vector<bool> > Y) {
            vector<vector<double> > y_prob = softmax(W * X + b));
            // error at last layer
            vector<vector<double> > delta = y_prob - Y;
            // return gradient of cros entropy cost
            return make_pair(delta * transpose(X), delta);
        }

        void train(vector<vector<double> > X, vector<vector<bool> > Y, int epochs=10, int batch_size=100, double lr=0.1) {
            for(int epoch=0;epoch<epochs;epoch++) {
                for(int i=0;i<n_rows(X);i+=batch_size) {
                    
                    pair<vector<vector<double> >, vector<double> > deltaJ;
                    deltaJ = grad(chunk(X, i, i+batch_size), chunk(Y, i, i+batch_size);
                    W = W - lr * deltaJ.first;
                    b = b - lr * deltaJ.second;
                }
            }
        }

};
