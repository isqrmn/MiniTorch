#pragma once

#include "BaseOpt.h"

class SGD : public OptimizerClass{
    DTYPE lr;

public:
    SGD(DTYPE lr=1e-3) : lr(lr){}

    DTYPE forward(PTR_E x) override {
        return x->getData() - x->getGradient() * lr;
    }

    DTYPE operator()(PTR_E x) override { return forward(x); }

    void step(const PTR_E &x, int i=0) override {
        if (x->getParam()) {
            auto temp = x->Copy();

            if (i >= opts.size()) {
                const auto &selected_optim = opts[i];
                x->setData(selected_optim->forward(temp));
            }
            else {
                this->opts.emplace_back(std::make_shared<SGD>(lr));
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
