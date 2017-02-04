#include "utils.cpp"
#include <vector>
#include <utility>
using namespace std;

typedef pair<Matrix, Vector> Gradient;

class NN {

    Matrix W;
    Vector b;

    public:

        pair<Gradient, double> grad(Matrix X, Matrix Y) {
            Matrix y_prob = softmax(W * X + b));
            // error at last layer
            Matrix delta = Y_prob - Y;
            
            // return gradient of cros entropy cost
            auto gradient = make_pair(delta * transpose(X), delta);
            double cost = CE(Y, Y_prob);

            return make_pair(gradient, cost);
        }

        Vector train(Matrix X, Matrix Y, int epochs=10, int batch_size=100, double lr=0.1) {
            Vector cost_history;

            for(int epoch=0; epoch<epochs; epoch++) {
                for(int i=0; i<n_rows(X); i+=batch_size) {
                    
                    auto res = grad(chunk(X, i, i+batch_size), chunk(Y, i, i+batch_size);
                    auto gradient = res.first;

                    W = W - lr * 1/batch_size * gradient.first;
                    b = b - lr * 1/batch_size * gradient.second;

                    auto cost = res.second;
                    cost_history.push_back(cost);
                }
            }

            return cost_history;
        }

};
