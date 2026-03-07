#pragma once

#include "../primitive/Linear.h"
#include "../primitive/activations/Tanh.h"

class RNN {
    bool bias = true;

    LLI F_IN;
    LLI HIDDEN_DIM;

    Linear x_linear;

    Tanh tanh_ = Tanh();

public:
    explicit RNN(LLI F_IN, LLI HIDDEN_DIM, const bool bias) : F_IN(F_IN), HIDDEN_DIM(HIDDEN_DIM), bias(bias), x_linear(Linear(F_IN + HIDDEN_DIM, HIDDEN_DIM)) {}

    explicit RNN() = delete;

    int getFin() { return F_IN; }

    int getHiddenDim() { return HIDDEN_DIM; }

    PTR_T getWeights() { return x_linear.getWeights(); }

    PTR_T getBias() { return x_linear.getBias(); }

    void setXWeights(const PTR_T &weights){ x_linear.setWeights(weights); }

    void setXBias(const PTR_T &bias){ x_linear.setBias(bias); }

    PTR_T forward(PTR_T x){
        // X : (B, C, L, DIM) -> (BC, L, DIM)
        // Hidden : (B, C, 1, DIM) -> (BC, DIM)
        const VEC_I &x_shape = x->getShape();

        const int BC = x_shape[x_shape.size() - 3] * x_shape[x_shape.size() - 4];

        const int L = x->getData().size() / BC / F_IN;

        PTR_T hidden = Minitorch::Zeros(BC * HIDDEN_DIM); // -> (HDIM * BC)
        hidden = Minitorch::ReShape(hidden, {1, 1, BC, HIDDEN_DIM});// -> (1, BC, HDIM)

        x = Minitorch::ReShape(x, {1, BC, L, F_IN}); // -> (BC, L, XDIM)
        x = Minitorch::TakeTranspose(x, 2, 1); // -> (L, BC, XDIM)

        // r_layer : (HDIM + XDIM) -> HDIM
        // x_layer : (HDIM + XDIM) -> HDIM
        // z_layer : (HDIM + XDIM) -> HDIM

        for (int i = 0; i < L; ++i){
                const VEC_I start = {0, i, 0, 0};
                const VEC_I end = {0, i, BC - 1, F_IN - 1};

                PTR_T input_flow = Minitorch::TakeSlice(x, start, end); // -> (1, BC, XDIM)

                auto catted_rz_input_flow = Minitorch::Concatenate(hidden, input_flow, 3); // -> (1, BC, HDIM + XDIM)

                auto x_gate = x_linear(catted_rz_input_flow); // -> (1, BC, HDIM + XDIM) -> (1, BC, HDIM)
                x_gate = tanh_(x_gate);

                auto final_hidden = Minitorch::FlexibleAdd(hidden, x_gate); // == (hidden * z_gate) + (x_gate * (1 - z_gate))

                hidden = final_hidden;
        }

        hidden = Minitorch::TakeTranspose(hidden, 2, 1); // -> (1, 1, BC, HDIM) -> (1, BC, 1, HDIM)

        return hidden;
    }

    PTR_T operator()(const PTR_T &x) { return this->forward(x); }
};
