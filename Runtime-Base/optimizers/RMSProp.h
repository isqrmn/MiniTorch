#pragma once
#include <memory>

#include "../../Runtime-Base/R_naming_cfg.h"

class RMSProp: public OptimizerClass{
    DTYPE one = 1e-0;
    DTYPE epsilon = 1e-6;

    DTYPE momentum_alpha = 1e-1;
    DTYPE momentum_alpha_2 = 0.9;

    DTYPE momentum = .0;

    DTYPE lr;

public:
    explicit RMSProp(DTYPE lr=1e-3) : lr(lr){}

    DTYPE forward(PTR_E x) override {
        DTYPE momentum_new_dir = std::pow(x->getGradient(), 2) * this->momentum_alpha;
        DTYPE momentum_old_dir = this->momentum * this->momentum_alpha_2;

        DTYPE new_momentum = momentum_new_dir + momentum_old_dir;

        DTYPE sqrt_momentum = std::sqrt(new_momentum + this->epsilon);

        DTYPE scaled_grad = x->getGradient() / sqrt_momentum;
        scaled_grad = scaled_grad * lr;

        momentum = new_momentum;

        return x->getData() - scaled_grad;
    }

    DTYPE operator()(PTR_E x) override { return forward(x); }

    void step(const PTR_E &x, int i=0) override {
        if (x->getParam()) {
            auto temp = x->Copy();

            if (i < opts.size()) {
                const auto &selected_optim = opts[i];
                x->setData(selected_optim->forward(temp));
            }
            else {
                this->opts.emplace_back(std::make_shared<RMSProp>(lr));
                const auto &selected_optim = opts[i];
                x->setData(selected_optim->forward(temp));
            }
        }
        x->setGradient(.0);

        if (x->getBackPath0() != nullptr) {
            step(x->getBackPath0(), ++i);
        }

        if (x->getBackPath1() != nullptr) {
            step(x->getBackPath1(), ++i);
        }
    }
};
