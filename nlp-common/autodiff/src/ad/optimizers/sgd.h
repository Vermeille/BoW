#pragma once

#include "../optimizer.h"

namespace ad {

namespace opt {

class SGD : public ad::Optimizer {
    public:
        SGD(float alpha) : alpha_(alpha) {}

        virtual void Update(ad::Var& v) {
            v.value() -= alpha_ * v.derivative();
        }

    private:
        float alpha_;
};

} // opt

} // ad
