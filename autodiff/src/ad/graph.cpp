#include "graph.h"

namespace ad {

const Var no_operand(
        new VarImpl(nullptr, Eigen::MatrixXd(1,1), -1, -1, -1, DoNothingBackprop));
void DoNothingBackprop(Var&, Var*, Var*) {}

Var ComputationGraph::CreateParam(std::shared_ptr<Eigen::MatrixXd> val) {
    int p_id = values_.size();

    values_.emplace_back(
            new VarImpl(this, val, p_id, -1, -1, DoNothingBackprop));
    return Var(values_.back().get());
}

Var ComputationGraph::CreateParam(const Eigen::MatrixXd& val) {
    return CreateParam(std::make_shared<Eigen::MatrixXd>(val));
}

Var ComputationGraph::CreateNode(
        const Eigen::MatrixXd& val,
        const Var& lhs,
        const Var& rhs,
        backward_t bwd) {
    int p_id = values_.size();

    values_.emplace_back(
            new VarImpl(this, val, p_id, lhs.id(), rhs.id(), *bwd));
    return Var(values_.back().get());
}

void ComputationGraph::BackpropFrom(Var& x) {
    int id = x.id();
    values_[id]->InitBackprop();
    for (int i = id; i >= 0; --i) {
        Var cur(values_[i].get());
        Var nullvar(nullptr);
        if (cur.lhs() == -1) {
            cur.Backward(nullptr, nullptr);
        } else if (cur.rhs() == -1) {
            Var a(values_[cur.lhs()].get());
            cur.Backward(&a, nullptr);
        } else {
            Var a(values_[cur.lhs()].get());
            Var b(values_[cur.rhs()].get());
            cur.Backward(&a, &b);
        }
    }
}

void ComputationGraph::ClearGrad() {
    for (auto& v : values_) {
        v->ClearDerivative();
    }
}

void ComputationGraph::Update(Optimizer& opt, const std::vector<Var*>& params) {
    for (auto& p : params) {
        opt.Update(*p);
    }
}

} // ad
