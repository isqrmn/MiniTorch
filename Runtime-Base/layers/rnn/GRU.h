#pragma once

#include "../primitive/Linear.h"
#include "../primitive/activations/Sigmoid.h"
#include "../primitive/activations/Tanh.h"

class GRU {
    // For Naming Convention I Used This Image: https://d2l.ai/chapter_recurrent-modern/gru.html

    LLI F_IN;
    LLI HIDDEN_DIM;

    bool bias = true;

    Linear x_linear;
    Linear r_linear;
    Linear z_linear;

    Tanh tanh_ = Tanh();
    Sigmoid sigmoid_ = Sigmoid();

public:
    explicit GRU(LLI F_IN, LLI HIDDEN_DIM, const bool bias=true) : F_IN(F_IN), HIDDEN_DIM(HIDDEN_DIM),
        bias(bias), x_linear(Linear(F_IN + HIDDEN_DIM, HIDDEN_DIM)), r_linear(Linear(F_IN + HIDDEN_DIM, HIDDEN_DIM)),
        z_linear(Linear(F_IN + HIDDEN_DIM, HIDDEN_DIM)) {}

    explicit GRU() = delete;

    int getFin() { return F_IN; }

    int getHiddenDim() { return HIDDEN_DIM; }

    PTR_T getXWeights() { return x_linear.getWeights(); }

    PTR_T getRWeights() { return r_linear.getWeights(); }

    PTR_T getZWeights() { return z_linear.getWeights(); }

    PTR_T getXBias() { return x_linear.getBias(); }

    PTR_T getRBias() { return r_linear.getBias(); }

    PTR_T getZBias() { return z_linear.getBias(); }

    void setXWeights(const PTR_T &weights){ x_linear.setWeights(weights); }

    void setRWeights(const PTR_T &weights){ r_linear.setWeights(weights); }

    void setZWeights(const PTR_T &weights){ z_linear.setWeights(weights); }

    void setXBias(const PTR_T &bias){ x_linear.setBias(bias); }

    void setRBias(const PTR_T &bias){ r_linear.setBias(bias); }

    void setZBias(const PTR_T &bias){ z_linear.setBias(bias); }

    // X : {1, B, L, D}
    PTR_T forward(PTR_T x) {
        // X : (B, C, L, DIM) -> (BC, L, DIM)
        // Hidden : (B, C, 1, DIM) -> (BC, DIM)
        const VEC_I &x_shape = x->getShape();

        const int BC = x_shape[x_shape.size() - 3] * x_shape[x_shape.size() - 4];

        const int L = x->getData().size() / BC / F_IN;

        PTR_T hidden = Minitorch::Zeros(BC * HIDDEN_DIM); // -> (HDIM * BC)
        hidden = Minitorch::ReShape(hidden, {1, 1, BC, HIDDEN_DIM});// -> (1, 1, BC, HDIM)

        x = Minitorch::ReShape(x, {1, BC, L, x_shape[x_shape.size() - 1]}); // -> (1, BC, L, XDIM)
        x = Minitorch::TakeTranspose(x, 2, 1); // -> (1, L, BC, XDIM)

        // r_layer : (HDIM + XDIM) -> HDIM
        // x_layer : (HDIM + XDIM) -> HDIM
        // z_layer : (HDIM + XDIM) -> HDIM

        for (int i = 0; i < L; ++i){
            const VEC_I start = {0, i, 0, 0};
            const VEC_I end = {0, i, BC - 1, F_IN - 1};

            PTR_T input_flow = Minitorch::TakeSlice(x, start, end); // -> (1, 1, BC, XDIM)

            auto catted_rz_input_flow = Minitorch::Concatenate(hidden, input_flow, 3); // -> (1, 1, BC, HDIM + XDIM)

            auto r_gate = r_linear(catted_rz_input_flow); // -> (1, 1, BC, HDIM + XDIM) -> (1, 1, BC, HDIM)
            r_gate = sigmoid_(r_gate);

            auto z_gate = z_linear(catted_rz_input_flow); // -> (1, 1, BC, HDIM + XDIM) -> (1, 1, BC, HDIM)
            z_gate = sigmoid_(z_gate); // -> scaler for update flow and hidden state

            auto resetted_hidden = Minitorch::FlexibleMul(hidden, r_gate); // (1, 1, BC, HDIM) * (1, 1, BC, HDIM)
            auto catted_x_input_flow = Minitorch::Concatenate(resetted_hidden, input_flow, 3); // -> (1, 1, BC, HDIM + XDIM)

            auto x_gate = x_linear(catted_x_input_flow); // -> (1, 1, BC, HDIM + XDIM) -> (1, 1, BC, HDIM)
            x_gate = tanh_(x_gate); // -> hidden state update flow

            auto z_gate_projected = Minitorch::FlexibleMul(z_gate, -1);
            z_gate_projected = Minitorch::FlexibleAdd(z_gate_projected, 1); // == 1 - z_gate

            auto scaled_x_gate = Minitorch::FlexibleMul(x_gate, z_gate_projected); // == x_gate * (1 - z_gate)

            auto scaled_hidden = Minitorch::FlexibleMul(hidden, z_gate); // == hidden * z_gate

            auto final_hidden = Minitorch::FlexibleAdd(scaled_hidden, scaled_x_gate); // == (hidden * z_gate) + (x_gate * (1 - z_gate))

            hidden = final_hidden;
        }

        hidden = Minitorch::TakeTranspose(hidden, 2, 1); // -> (1, 1, BC, HDIM) -> (1, BC, 1, HDIM)

        return hidden;
    }

    PTR_T operator()(const PTR_T &x) { return this->forward(x); }
};
