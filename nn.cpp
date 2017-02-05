#include "utils.cpp"
#include <vector>
#include <utility>
using namespace std;

class NN {

    Matrix W;
    Vector b;

    public:

        NN(int num_classes=2, int data_dim=3) {
            // b = new Vector(num_classes);
            W = blank_matrix(data_dim, num_classes, 0.);            
        }

        double grad(Matrix X, Matrix Y, Matrix &grad_W, Vector &grad_b) {
            Matrix Y_prob = softmax(X * W);
            // error at last layer
            Matrix delta = Y_prob - Y;
            
            // return gradient of cros entropy cost
            grad_W = delta * X;
            // grad_b = delta;
            double cost = CE(Y, Y_prob);
            return 1.;
        }

        Vector train(Matrix X, Matrix Y, int epochs=10, int batch_size=100, double lr=0.1) {
            Vector cost_history;
            // X = transpose(X);
            print(X);
            Y = transpose(Y);            
            Matrix grad_W = blank_matrix(3, 2, 0.); 
            Vector grad_b;

            for(int epoch=0; epoch<epochs; epoch++) {
                for(int i=0; i<n_rows(X); i+=batch_size) {
                    
                    auto cost = grad(chunk(X, i, i+batch_size), chunk(Y, i, i+batch_size), grad_W, grad_b);

                    W = W - lr * 1/batch_size * grad_W;
                    // b = b - lr * 1/batch_size * grad_b;

                    cost_history.push_back(cost);
                }
            }

            return cost_history;
        }

};
