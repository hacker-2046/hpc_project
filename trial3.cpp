#include "rnn.h"
#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <xtensor-blas/xlinalg.hpp>

int main(int argc, char **argv){
    RNN rnn(5, 3, 10);
    auto x = xt::random::rand<double>({500, 20, 5});
    auto y = xt::random::rand<double>({500, 3});

    double lr = 0.1; // Adjust learning rate
    int iterations = atoi(argv[1]); // Adjust iterations

    for(int i=0; i<iterations ;i++){
        auto out = rnn.forward(x);
        auto err = (y-out);
        auto loss = xt::mean(xt::square(err));
        //std::cout << xt::view(out, xt::range(0, 3), xt::all()) << std::endl;
        //std::cout << xt::view(y, xt::range(0, 3), xt::all()) << std::endl;
        std::cout << "epoch: " << i << " loss: " << loss << "\n";
        auto next = rnn.backward(err, lr*i/iterations);
    }

    std::cout << xt::view(y, xt::range(0, 4), xt::all()) << std::endl;
    std::cout << xt::view(rnn.forward(x), xt::range(0, 4), xt::all()) << std::endl;
}
