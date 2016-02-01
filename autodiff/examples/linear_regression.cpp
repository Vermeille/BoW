#include <vector>
#include <memory>
#include <iostream>
#include <cmath>

#include <ad/ad.h>

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> GenDataset(int nb_examples) {
    // dataset
    Eigen::MatrixXd x(2, nb_examples);
    Eigen::MatrixXd y(1, nb_examples);
    for (int k = 0; k < nb_examples; ++k) {
        float x1 = (rand() % 30 - 15) / 15.;
        float x2 = (rand() % 30 - 15) / 15.;
        x(0, k) = x1;
        x(1, k) = x2;
        y(0, k) =  x1 * 8 + x2 * 3 + 5;
    }
    return std::make_pair(x, y);
}

int main() {
    using namespace ad;
    const int nb_examples = 1;

    auto a_weights = std::make_shared<Eigen::MatrixXd>(1, 2);
    *a_weights << 3, 4;
    auto b_weights = std::make_shared<Eigen::MatrixXd>(1, 1);
    *b_weights << 6;

    for (int i = 0; i < 100; ++ i) {
        ComputationGraph g;
        auto dataset = GenDataset(nb_examples);

        Var x = g.CreateParam(dataset.first);
        Var y = g.CreateParam(dataset.second);
        Var a = g.CreateParam(a_weights);
        Var b = g.CreateParam(b_weights);

        Var h = a * x + b;
        Var j = MSE(h, y);

        std::cout << "COST = " << j.value() << "\n";

        opt::SGD sgd(0.1 / nb_examples);
        g.BackpropFrom(j);
        g.Update(sgd, {&a, &b});
    }

    std::cout << "a = " << *a_weights << " b = " << *b_weights << std::endl;
    return 0;
}

