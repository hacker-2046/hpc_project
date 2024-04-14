//g++ trial2.cpp -I xtensor/include -I xtl/include -I xtensor-blas/include
#include "linear.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <xtensor-blas/xlinalg.hpp>

int main(int argc, char **argv) {
    Linear lin(500, 300, "None");
    xt::xarray<double> x = xt::random::randn<double>({1000,500});

    xt::xarray<double> w = xt::random::randn<double>({500, 300}); 
    
    auto y = xt::linalg::dot(x, w) + 123;

    double lr = 0.1; // Adjust learning rate
    int iterations = atoi(argv[1]); // Adjust iterations

    for(int i=0; i<iterations ;i++){
        auto out = lin.forward(x);
        auto err = (y-out);
        auto loss = xt::mean(xt::square(err));
        std::cout << "epoch: " << i << " loss: " << loss << "\n";
        auto next = lin.backward(x, out, err, lr);
    }
    std::cout << "Weights:\n" << lin.weights << std::endl;
    std::cout << "Bias:\n" << lin.bias << std::endl;
}
