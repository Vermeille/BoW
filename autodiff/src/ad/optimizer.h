#pragma once

namespace ad {

class Var;

class Optimizer {
    public:
        virtual void Update(Var& v) = 0;
        virtual void NextIteration() {}
};

}
