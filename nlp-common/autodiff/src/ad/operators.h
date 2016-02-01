#pragma once

#include <memory>

#include "graph.h"

namespace ad {

Var operator+(const Var& v1, const Var& v2);
Var operator-(const Var& v1, const Var& v2);
Var operator*(const Var& v1, const Var& v2);
Var Relu(const Var& v1);
Var Square(const Var& v1);
Var EltSquare(const Var& v1);
Var operator^(const Var& v1, const Var& v2);
Var Log(const Var& x);
Var NLog(const Var& x);
Var CrossEntropy(const Var& y, const Var& h);
Var Exp(const Var& x);
Var Softmax(const Var& x);
Var Sigmoid(const Var& x);
Var Sum(const Var& a);
Var MSE(const Var& h, const Var& y);

}
