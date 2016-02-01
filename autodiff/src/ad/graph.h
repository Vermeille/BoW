#pragma once

#include <list>
#include <unordered_map>
#include <memory>

#include "optimizer.h"

#include "Eigen/Dense"

namespace ad {

class Var;
class ComputationGraph;

using backward_t = void(*)(Var&, Var*, Var*);
void DoNothingBackprop(Var&, Var*, Var*);

class VarImpl {
    private:
        std::shared_ptr<Eigen::MatrixXd> value_;
        Eigen::MatrixXd derivative_;

        int lhs_;
        int rhs_;
        int id_;

        backward_t backward_;

        ComputationGraph* const graph_;

    public:
        VarImpl(ComputationGraph* g,
                std::shared_ptr<Eigen::MatrixXd> val,
                int my_id,
                int op1,
                int op2,
                const backward_t& bckwd)
            : value_(val), derivative_(val->rows(), val->cols()), lhs_(op1),
            rhs_(op2), id_(my_id), backward_(bckwd), graph_(g) {
            derivative_.setZero();
        }

        VarImpl(ComputationGraph* g,
                const Eigen::MatrixXd& val,
                int my_id,
                int op1,
                int op2,
                const backward_t& bckwd)
            : value_(std::make_shared<Eigen::MatrixXd>(val)),
            derivative_(val.rows(), val.cols()), lhs_(op1),
            rhs_(op2), id_(my_id), backward_(bckwd), graph_(g) {
            derivative_.setZero();
        }
        ComputationGraph* graph() const { return graph_; }
        const Eigen::MatrixXd& value() const { return *value_;}
        Eigen::MatrixXd& value() { return *value_;}
        const Eigen::MatrixXd& derivative() const { return derivative_;}
        Eigen::MatrixXd& derivative() { return derivative_;}

        void ClearDerivative() { derivative_.setZero(); }

        void Backward(Var& self, Var* lhs, Var* rhs) {
            backward_(self, lhs, rhs);
        }

        void InitBackprop() { derivative_.setOnes(); }

        int id() const { return id_; }
        int lhs() const { return lhs_; }
        int rhs() const { return rhs_; }
};

class Var {
    VarImpl* var_;
    public:
        Var(VarImpl* var) : var_(var) {}
        ComputationGraph* graph() const { return var_->graph(); }
        const Eigen::MatrixXd& value() const { return var_->value();}
        Eigen::MatrixXd& value() { return var_->value();}
        const Eigen::MatrixXd& derivative() const { return var_->derivative();}
        Eigen::MatrixXd& derivative() { return var_->derivative();}

        void Backward(Var* lhs, Var* rhs) {
            var_->Backward(*this, lhs, rhs);
        }
        int id() const { return var_->id(); }
        int lhs() const { return var_->lhs(); }
        int rhs() const { return var_->rhs(); }
};

extern const Var no_operand;

class ComputationGraph {
    std::vector<std::unique_ptr<VarImpl>> values_;

    public:
    Var CreateParam(std::shared_ptr<Eigen::MatrixXd> val);
    Var CreateParam(const Eigen::MatrixXd& val);
    Var CreateNode(
            const Eigen::MatrixXd& val,
            const Var& lhs,
            const Var& rhs,
            backward_t bwd);
    void BackpropFrom(Var& x);
    void ClearGrad();
    void Update(Optimizer& opt, const std::vector<Var*>& params);
};

}

