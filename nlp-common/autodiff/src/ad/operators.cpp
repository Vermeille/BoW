#include <cassert>

#include "operators.h"

namespace ad {

static void AddBackprop(Var& val, Var* lhs, Var* rhs) {
    lhs->derivative() += val.derivative();
    rhs->derivative() += val.derivative();
}

Var operator+(const Var& v1, const Var& v2) {
    return v1.graph()->CreateNode(v1.value() + v2.value(), v1, v2, AddBackprop);
}

static void SubBackprop(Var& val, Var* lhs, Var* rhs) {
    lhs->derivative() += val.derivative();
    rhs->derivative() -= val.derivative();
}

Var operator-(const Var& v1, const Var& v2) {
    return v1.graph()->CreateNode(v1.value() - v2.value(), v1, v2, SubBackprop);
}

static void MulBackprop(Var& val, Var* lhs, Var* rhs) {
    lhs->derivative() +=  val.derivative() * rhs->value().transpose();
    rhs->derivative() +=  lhs->value().transpose() * val.derivative();
}

Var operator*(const Var& v1, const Var& v2) {
    return v1.graph()->CreateNode(v1.value() * v2.value(), v1, v2, MulBackprop);
}

static void CoeffMulBackprop(Var& val, Var* lhs, Var* rhs) {
    lhs->derivative() +=  val.derivative() * rhs->value()(0, 0);
}

Var operator*(double a, const Var& v1) {
    Eigen::MatrixXd coeff(1, 1);
    coeff << a;
    Var coeff_var = v1.graph()->CreateParam(coeff);
    return v1.graph()->CreateNode(v1.value() * a, v1, coeff_var, CoeffMulBackprop);
}

Var operator*(const Var& v1, double a) {
    Eigen::MatrixXd coeff(1, 1);
    coeff << a;
    Var coeff_var = v1.graph()->CreateParam(coeff);
    return v1.graph()->CreateNode(v1.value() * a, v1, coeff_var, CoeffMulBackprop);
}

static void ReluBackprop(Var& val, Var* lhs, Var*) {
    double* da = lhs->derivative().data();
    const double* a = lhs->value().data();
    double* db = val.derivative().data();
    for (int i = 0; i < lhs->derivative().size(); ++i) {
        da[i] += a[i] > 0 ? db[i] : 0;
    }
}

Var Relu(const Var& v1) {
    return v1.graph()->CreateNode(
            v1.value().array().max(0), v1, no_operand, ReluBackprop);
}

static void SquareBackprop(Var& val, Var* lhs, Var*) {
    lhs->derivative() += 2 * val.derivative() * lhs->value();
}

Var Square(const Var& v1) {
    return v1.graph()->CreateNode(
            v1.value() * v1.value(), v1, no_operand, SquareBackprop);
}

static void EltSquareBackprop(Var& val, Var* lhs, Var*) {
    lhs->derivative() += 2 * val.derivative().cwiseProduct(lhs->value());
}

Var EltSquare(const Var& v1) {
    return v1.graph()->CreateNode(
            v1.value().cwiseProduct(v1.value()), v1, no_operand, EltSquareBackprop);
}

static void EltwiseMulBackprop(Var& val, Var* lhs, Var* rhs) {
    lhs->derivative() += val.derivative().cwiseProduct(rhs->value());
    rhs->derivative() += val.derivative().cwiseProduct(lhs->value());
}

Var operator^(const Var& v1, const Var& v2) {
    return v1.graph()->CreateNode(
            v1.value().cwiseProduct(v2.value()), v1, v2, EltwiseMulBackprop);
}

static void LogBackprop(Var& val, Var* lhs, Var*) {
    double* da = lhs->derivative().data();
    double* dx = val.derivative().data();
    const double* a = lhs->value().data();
    for (int i = 0; i < val.value().size(); ++i) {
        da[i] += dx[i] / a[i];
    }
}

Var Log(const Var& x) {
    Eigen::MatrixXd res(x.value().rows(), x.value().cols());
    double* dst_ptr = res.data();
    const double* src_ptr = x.value().data();
    for (int i = 0; i < res.size(); ++i) {
        dst_ptr[i] = log(src_ptr[i]);
    }
    return x.graph()->CreateNode(res, x, no_operand, LogBackprop);
}

static void NLogBackprop(Var& val, Var* lhs, Var*) {
    double* da = lhs->derivative().data();
    double* dx = val.derivative().data();
    const double* a = lhs->value().data();
    for (int i = 0, size = val.value().size(); i < size; ++i) {
        da[i] += -dx[i] / (a[i]);
    }
}

Var NLog(const Var& x) {
    Eigen::MatrixXd res(x.value().rows(), x.value().cols());
    double* dst_ptr = res.data();
    const double* src_ptr = x.value().data();
    for (int i = 0; i < res.size(); ++i) {
        dst_ptr[i] = -log(src_ptr[i]);
    }
    return x.graph()->CreateNode(res, x, no_operand, NLogBackprop);
}

Var CrossEntropy(const Var& y, const Var& h) {
    return Sum(y ^ NLog(h));
}

static void ExpBackprop(Var& val, Var* lhs, Var*) {
    double* da = lhs->derivative().data();
    const double* dx = val.derivative().data();
    const double* a = lhs->value().data();
    for (int i = 0; i < val.value().size(); ++i) {
        da[i] += dx[i] * exp(a[i]);
    }
}

Var Exp(const Var& x) {
    Eigen::MatrixXd res(x.value().rows(), x.value().cols());
    double* dst_ptr = res.data();
    const double* src_ptr = x.value().data();
    for (int i = 0; i < res.size(); ++i) {
        dst_ptr[i] = exp(src_ptr[i]);
    }
    return x.graph()->CreateNode(res, x, no_operand, ExpBackprop);
}

static void SoftmaxBackprop(Var& val, Var* lhs, Var*) {
    const Eigen::MatrixXd& a = val.value();
    lhs->derivative() += val.derivative()
        .cwiseProduct((a.array() * (1.0 - a.array())).matrix());
}

Var Softmax(const Var& x) {
    Eigen::MatrixXd res(x.value().rows(), x.value().cols());
    double* dst_ptr = res.data();
    const double* src_ptr = x.value().data();
    double total = 0;
    for (int i = 0; i < res.size(); ++i) {
        dst_ptr[i] = exp(src_ptr[i]);
        total += dst_ptr[i];
    }

    for (int i = 0; i < res.size(); ++i) {
        dst_ptr[i] /= total;
    }
    return x.graph()->CreateNode(res, x, no_operand, SoftmaxBackprop);
}

Var Sigmoid(const Var& x) {
    Eigen::MatrixXd res(x.value().rows(), x.value().cols());
    double* dst_ptr = res.data();
    const double* src_ptr = x.value().data();
    for (int i = 0; i < res.size(); ++i) {
        dst_ptr[i] = 1.0 / (1 + exp(-src_ptr[i]));
    }

    // Sigmoid derivative is the same as Softmax's
    return x.graph()->CreateNode(res, x, no_operand, SoftmaxBackprop);
}

void SumBackprop(Var& val, Var* lhs, Var*) {
    lhs->derivative().array() += (double)val.derivative()(0, 0);
}

Var Sum(const Var& a) {
    Eigen::MatrixXd res(1, 1);
    res << a.value().sum();
    return a.graph()->CreateNode(res, a, no_operand, SumBackprop);
}

void MeanBackprop(Var& val, Var* lhs, Var*) {
    lhs->derivative().array() +=
        (double)val.derivative()(0, 0) / val.value().size();
}

Var Mean(const Var& a) {
    Eigen::MatrixXd res(1, 1);
    res << a.value().sum() / a.value().size();
    return a.graph()->CreateNode(res, a, no_operand, MeanBackprop);
}

Var MSE(const Var& h, const Var& y) {
    return Sum(EltSquare(h - y));
}

}
