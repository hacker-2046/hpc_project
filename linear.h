// linear.h
#ifndef LINEAR_H
#define LINEAR_H

#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xoperation.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <string>

class Linear {
public:
    xt::xarray<double> weights;
    xt::xarray<double> bias;
    std::string activation;

    Linear(int input_size, int output_size, std::string act = "relu") :
        activation(act),
        weights(xt::random::randn<double>({input_size, output_size})),
        bias(xt::random::randn<double>({output_size})) {}

    xt::xarray<double> relu(const xt::xarray<double>& input);
    xt::xarray<double> relu_gradient(const xt::xarray<double>& input);
    xt::xarray<double> sigmoid(const xt::xarray<double>& input);
    xt::xarray<double> sigmoid_gradient(const xt::xarray<double>& input);

    xt::xarray<double> forward(const xt::xarray<double>& input);
    xt::xarray<double> backward(
        const xt::xarray<double>& input,
        const xt::xarray<double>& output, 
        const xt::xarray<double>& error,
        double lr = 0.01);
};

inline xt::xarray<double> Linear::forward(const xt::xarray<double>& input) {
    xt::xarray<double> output = xt::linalg::dot(input, weights) + bias;
    if (activation == "relu") output = relu(output);
    else if (activation == "sigmoid") output = sigmoid(output);
    return output;
}

inline xt::xarray<double> Linear::backward(
        const xt::xarray<double>& input,
        const xt::xarray<double>& output, 
        const xt::xarray<double>& error,
        double lr) {
    xt::xarray<double> delta = error;
    if (activation == "relu") delta *= relu_gradient(output);
    else if (activation == "sigmoid") delta *= sigmoid_gradient(output);

    xt::xarray<double> delta_w = xt::transpose(xt::linalg::dot(xt::transpose(delta), input));
    xt::xarray<double> delta_b = xt::sum(delta, {0});
    xt::xarray<double> next_error = xt::linalg::dot(delta, xt::transpose(weights));

    int batch_size = input.shape()[0];

    weights += delta_w*lr / batch_size;
    bias += delta_b*lr / batch_size;

    return next_error;
}

inline xt::xarray<double> Linear::relu(const xt::xarray<double>& input) {
    return input * (input >= 0.0);
}

inline xt::xarray<double> Linear::relu_gradient(const xt::xarray<double>& input) {
    return (input >= 0.0);
}

inline xt::xarray<double> Linear::sigmoid(const xt::xarray<double>& input) {
    return 1 / (1 + xt::exp(-input));
}

inline xt::xarray<double> Linear::sigmoid_gradient(const xt::xarray<double>& input) {
    xt::xarray<double> sigmoid_output = sigmoid(input);
    return sigmoid_output * (1 - sigmoid_output);
}

#endif /* LINEAR_H */
