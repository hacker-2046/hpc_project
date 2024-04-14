//g++ trial2.cpp -I xtensor/include -I xtl/include -I xtensor-blas/include
#include "linear.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <xtensor-blas/xlinalg.hpp>

int main(int argc, char **argv) {
    Linear lin(5, 2, "None");
    xt::xarray<double> x = xt::random::randn<double>({100,5});

    xt::xarray<double> w = {{1,27,3,41,1},
                            {45,5,6,500,3}};
    
    auto y = xt::linalg::dot(x, xt::transpose(w)) + 123;

    double lr = 0.1; // Adjust learning rate
    int iterations = atoi(argv[1]); // Adjust iterations

    for(int i=0; i<iterations ;i++){
        auto out = lin.forward(x);
        auto err = (y-out);
        auto loss = xt::mean(xt::square(err));
        if(!(i%100)){
            std::cout << "epoch: " << i << " loss: " << loss << "\n";
        }
        auto next = lin.backward(x, out, err, lr);
    }
    std::cout << "Weights:\n" << lin.weights << std::endl;
    std::cout << "Bias:\n" << lin.bias << std::endl;
}
