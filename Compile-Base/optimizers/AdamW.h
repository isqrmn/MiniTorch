#pragma once

class AdamW : public OptimizerClass{
    DTYPE one = 1e-0;
    DTYPE epsilon = 1e-6;

    DTYPE momentum_beta = 1e-1;
    DTYPE momentum_beta_2 = 0.9;

    DTYPE momentum_alpha = 1e-2;
    DTYPE momentum_alpha_2 = 0.99;

    DTYPE first_momentum = .0;
    DTYPE second_momentum = .0;

    DTYPE lr;

    DTYPE weight_decay;

    int t = 1;

public:
    explicit AdamW(DTYPE lr=1e-3, DTYPE weight_decay=1e-6) : lr(lr), weight_decay(weight_decay) {}

    DTYPE forward(PTR_E x) override {
        DTYPE s_momentum_new_dir = std::pow(x->getGradient(), 2) * this->momentum_alpha;
        DTYPE s_momentum_old_dir = this->second_momentum * this->momentum_alpha_2;

        DTYPE new_momentum_s = s_momentum_new_dir + s_momentum_old_dir;
        new_momentum_s = new_momentum_s / (one - std::pow(momentum_alpha_2, t));

        DTYPE sqrt_momentum = std::sqrt(new_momentum_s + this->epsilon);

        DTYPE f_momentum_new_dir = (x->getGradient()) * this->momentum_beta;
        DTYPE f_momentum_old_dir = this->first_momentum * this->momentum_beta_2;

        DTYPE new_momentum_f = f_momentum_new_dir + f_momentum_old_dir;
        new_momentum_f = new_momentum_f / (one - std::pow(momentum_beta_2, t));

        DTYPE scaled_grad = new_momentum_f / sqrt_momentum;
        scaled_grad = scaled_grad * lr;

        first_momentum = new_momentum_f;
        second_momentum = new_momentum_s;

        if (t < 10000) {
            ++t;
        }

        DTYPE real_decay = weight_decay * x->getData();

        return x->getData() - scaled_grad - real_decay;
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
                this->opts.emplace_back(std::make_shared<Adam>(lr));
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
