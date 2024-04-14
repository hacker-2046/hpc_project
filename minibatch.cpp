// g++ mainCode.cpp -I xtensor/include -I xtl/include -I xtensor-blas/include
#include "linear.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <omp.h>

int main(int argc, char **argv) {
    Linear lin(500, 300, "None");
    int size = 1000;
    xt::xarray<double> x = xt::random::randn<double>({size, 500});

    xt::xarray<double> w = xt::random::randn<double>({500, 300}); 
    
    auto y = xt::linalg::dot(x, w) + 123;

    double lr = 0.1; // Adjust learning rate
    int iterations = atoi(argv[1]); // Adjust iterations

    int factor = size / 10;
    xt::xarray<double> all_outs = xt::zeros<double>({size, 300});
    
    for(int i = 0; i < iterations; i++) {
        #pragma omp parallel for
        for(int j = 0; j < size; j += factor) {
            auto inp = xt::view(x, xt::range(j, j+factor), xt::all());
            auto out = lin.forward(inp);
            xt::view(all_outs, xt::range(j, j + factor), xt::all()) = out;
        }

        auto err = (y - all_outs);
        auto loss = xt::mean(xt::square(err));
        std::cout << "epoch: [" << i << "] loss: " << loss << "\n";
        
        lin.backward(x, all_outs, err, lr);
    }

    //std::cout << "Weights:\n" << lin.weights << std::endl;
    //std::cout << "Bias:\n" << lin.bias << std::endl;
}
