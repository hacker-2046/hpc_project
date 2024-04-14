//rnn.h
#ifndef RNN_H
#define RNN_H

#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include "linear.h"
#include <omp.h>

class RNN {
private:
    int inp;
    int out;
    int hide;
    Linear linear_input;
    Linear linear_hidden;
    Linear linear_output;
    // for back propagation
    xt::xarray<double> input;
    xt::xarray<double> output;
    xt::xarray<double> hidden;

public:
    RNN(int inp, int out, int hide=100) :
        inp(inp), out(out), hide(hide),
        linear_input(inp, hide, "relu"),
        linear_hidden(hide, hide, "sigmoid"),
        linear_output(hide, out, "sigmoid") {}

    xt::xarray<double> forward(const xt::xarray<double>& x);
    xt::xarray<double> backward(const xt::xarray<double>& error, double lr);
};

inline xt::xarray<double> RNN::forward(const xt::xarray<double>& input) {
    // input should be a sequence of inputs (shape: N, seq, inp)
    this->input = input;
    int N = input.shape()[0];
    int seq = input.shape()[1];

    xt::xarray<double> x = xt::zeros<double>({N, seq, hide});
    #pragma omp parallel for
    for (int i = 0; i < seq; i++){
        xt::view(x, xt::all(), i, xt::all()) = linear_input.forward(xt::view(input, xt::all(), i, xt::all()));
    }

    hidden = xt::zeros<double>({N, seq+1, hide});
    xt::xarray<double> last_hidden;
    for (int i = 0; i < seq; i++) {
        last_hidden = xt::view(hidden, xt::all(), i, xt::all());
        xt::view(hidden, xt::all(), i+1, xt::all()) = xt::view(x, xt::all(), i, xt::all()) + linear_hidden.forward(last_hidden);
    }

    output = linear_output.forward(xt::view(hidden, xt::all(), -1, xt::all()));
    return output;
}

inline xt::xarray<double> RNN::backward(const xt::xarray<double>& error, double lr) {
    int N = input.shape()[0];
    int seq = input.shape()[1];

    xt::xarray<double> output_grad = linear_output.backward(xt::view(hidden, xt::all(), -1, xt::all()), output, error, lr);
    xt::xarray<double> delta_hidden = xt::zeros<double>({N, seq + 1, hide});
    xt::view(delta_hidden, xt::all(), -1, xt::all()) = output_grad;

    xt::xarray<double> current_output_grad, delta_hidden_current;
    #pragma omp parallel for
    for (int i = seq - 1; i >= 0; i--) {
        xt::xarray<double> current_hidden = xt::view(hidden, xt::all(), i + 1, xt::all());
        xt::xarray<double> last_hidden = xt::view(hidden, xt::all(), i, xt::all());
        
        #pragma omp critical
        {
            current_output_grad = xt::view(delta_hidden, xt::all(), i + 1, xt::all());
            delta_hidden_current = linear_hidden.backward(last_hidden, current_hidden, current_output_grad, lr);
            xt::view(delta_hidden, xt::all(), i, xt::all()) = delta_hidden_current;
        }
    }
    
    /* parallel input layer update */
    xt::xarray<double> delta_input = xt::zeros<double>({N, seq, inp});
    #pragma omp parallel for
    for (int i = 0; i < seq; i++) {
        xt::xarray<double> current_input = xt::view(input, xt::all(), i, xt::all());
        xt::xarray<double> current_error = xt::view(delta_hidden, xt::all(), i + 1, xt::all());

        #pragma omp critical
        {
            xt::view(delta_input, xt::all(), i, xt::all()) = 
            linear_input.backward(
                current_input,  // current input
                current_error,  // current error
                lr
            );
        }
    }

    return delta_input;
}
#endif /* RNN_H */
