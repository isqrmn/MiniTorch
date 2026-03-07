#pragma once

class Linear {
    const bool bias = true;

    LLI F_IN;
    LLI F_OUT;

    PTR_T weights;
    PTR_T bias_weight;

    void Initialization() {
        PTR_T weights = Minitorch::HeUniformTensorInitialization(F_OUT * F_IN, F_IN);

        const VEC_I w_shape{1, 1, F_IN, F_OUT};
        weights = Minitorch::ReShape(weights, w_shape);

        setWeights(weights);

        if (bias) {
            PTR_T bias = Minitorch::HeUniformTensorInitialization(F_OUT, F_OUT);

            const VEC_I b_shape{1, 1, 1, F_OUT};
            bias = Minitorch::ReShape(bias, b_shape);

            setBias(bias);
        }
    }

public:
    explicit Linear(LLI F_IN, LLI F_OUT, const bool bias=true): bias(bias), F_IN(F_IN), F_OUT(F_OUT) {
        Initialization();
    }

    explicit Linear() = delete;

    PTR_T getWeights(){ return weights; }

    PTR_T getBias(){ return bias_weight; }

    void setWeights(const PTR_T &weights){ this->weights = weights; }

    void setBias(const PTR_T &bias){ this->bias_weight = bias; }

    PTR_T forward(const PTR_T &x) const{
        PTR_T out = Minitorch::MatMul(x, weights);
        if (bias) {
            out = Minitorch::FlexibleAdd(out, bias_weight);
        }

        return out;
    }

    PTR_T operator()(const PTR_T &x) const { return forward(x); }
};
