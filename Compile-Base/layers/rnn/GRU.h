#pragma once

#include "../primitive/Linear.h"
#include "../primitive/activations/Sigmoid.h"
#include "../primitive/activations/Tanh.h"

template<int T, int BC, int F_IN, int HIDDEN_DIM>
class GRU {
    // For Naming Convention I Used This Image: https://d2l.ai/chapter_recurrent-modern/gru.html

    bool bias = true;

    Linear<BC * (F_IN + HIDDEN_DIM), BC, F_IN + HIDDEN_DIM, HIDDEN_DIM> x_linear = typename Linear<BC * (F_IN + HIDDEN_DIM), BC, F_IN + HIDDEN_DIM, HIDDEN_DIM>::Linear(bias);

    Linear<BC * (F_IN + HIDDEN_DIM), BC, F_IN + HIDDEN_DIM, HIDDEN_DIM> r_linear = typename Linear<BC * (F_IN + HIDDEN_DIM), BC, F_IN + HIDDEN_DIM, HIDDEN_DIM>::Linear(bias);

    Linear<BC * (F_IN + HIDDEN_DIM), BC, F_IN + HIDDEN_DIM, HIDDEN_DIM> z_linear = typename Linear<BC * (F_IN + HIDDEN_DIM), BC, F_IN + HIDDEN_DIM, HIDDEN_DIM>::Linear(bias);

    Tanh<BC * HIDDEN_DIM> tanh_ = typename Tanh<BC * HIDDEN_DIM>::Tanh();
    Sigmoid<BC * HIDDEN_DIM> sigmoid_ = typename Sigmoid<BC * HIDDEN_DIM>::Sigmoid();

public:
    explicit GRU(const bool bias) : bias(bias) {}

    explicit GRU() {}

    static int getFin() { return F_IN; }

    static int getHiddenDim() { return HIDDEN_DIM; }

    SPTR<Tensor<F_IN * HIDDEN_DIM>> getXWeights() const{ return x_linear->getWeights(); }

    SPTR<Tensor<F_IN * HIDDEN_DIM>> getRWeights() const{ return r_linear->getWeights(); }

    SPTR<Tensor<F_IN * HIDDEN_DIM>> getZWeights() const{ return z_linear->getWeights(); }

    SPTR<Tensor<HIDDEN_DIM>> getXBias() const{ return x_linear->getBias(); }

    SPTR<Tensor<HIDDEN_DIM>> getRBias() const{ return r_linear->getBias(); }

    SPTR<Tensor<HIDDEN_DIM>> getZBias() const{ return z_linear->getBias(); }

    void setXWeights(const SPTR<Tensor<(F_IN + HIDDEN_DIM) * HIDDEN_DIM>> &weights){ x_linear.setWeights(weights); }

    void setRWeights(const SPTR<Tensor<(F_IN + HIDDEN_DIM) * HIDDEN_DIM>> &weights){ r_linear.setWeights(weights); }

    void setZWeights(const SPTR<Tensor<(F_IN + HIDDEN_DIM) * HIDDEN_DIM>> &weights){ z_linear.setWeights(weights); }

    void setXBias(const SPTR<Tensor<HIDDEN_DIM>> &bias){ x_linear.setBias(bias); }

    void setRBias(const SPTR<Tensor<HIDDEN_DIM>> &bias){ r_linear.setBias(bias); }

    void setZBias(const SPTR<Tensor<HIDDEN_DIM>> &bias){ z_linear.setBias(bias); }

    SPTR<Tensor<HIDDEN_DIM * BC>> forward(SPTR<Tensor<T>> x){
        // X : (B, C, L, DIM) -> (BC, L, DIM)
        // Hidden : (B, C, 1, DIM) -> (BC, DIM)
        constexpr int L = T / BC / F_IN;
        constexpr int SIZE = F_IN * HIDDEN_DIM;

        const VEC_I &x_shape = x->getShape();

        SPTR<Tensor<BC * HIDDEN_DIM>> hidden = MiniTorch<BC * HIDDEN_DIM>::Zeros(); // -> (HDIM * BC)
        hidden = MiniTorch<BC * HIDDEN_DIM>::ReShape(hidden, {1, BC, HIDDEN_DIM});// -> (1, BC, HDIM)

        x = MiniTorch<T>::ReShape(x, {BC, L, x_shape[3]}); // -> (BC, L, XDIM)
        x = MiniTorch<T>::TakeTranspose(x, 1, 0); // -> (L, BC, XDIM)

        // r_layer : (HDIM + XDIM) -> HDIM
        // x_layer : (HDIM + XDIM) -> HDIM
        // z_layer : (HDIM + XDIM) -> HDIM

        auto loop = [&]<size_t... I>(std::index_sequence<I...>) {
            ([&]{
                constexpr ARR<int, 3> start = {I, 0, 0};
                constexpr ARR<int, 3> end = {I, BC - 1, F_IN - 1};

                SPTR<Tensor<BC * F_IN>> input_flow = MiniTorch<T>::template TakeSlice<3, BC * F_IN, start, end>(x); // -> (1, BC, XDIM)

                auto catted_rz_input_flow = MiniTorch<BC * HIDDEN_DIM>::template Concatenate<BC * F_IN>(hidden, input_flow, 2); // -> (1, BC, HDIM + XDIM)

                auto r_gate = r_linear(catted_rz_input_flow); // -> (1, BC, HDIM + XDIM) -> (1, BC, HDIM)
                r_gate = sigmoid_(r_gate);

                auto z_gate = z_linear(catted_rz_input_flow); // -> (1, BC, HDIM + XDIM) -> (1, BC, HDIM)
                z_gate = sigmoid_(z_gate); // -> scaler for update flow and hidden state

                auto resetted_hidden = MiniTorch<BC * HIDDEN_DIM>::template FlexibleMul<BC * HIDDEN_DIM>(hidden, r_gate); // (1, BC, HDIM) * (1, BC, HDIM)
                auto catted_x_input_flow = MiniTorch<BC * HIDDEN_DIM>::template Concatenate<BC * F_IN>(resetted_hidden, input_flow, 2); // -> (1, BC, HDIM + XDIM)

                auto x_gate = x_linear(catted_x_input_flow); // -> (1, BC, HDIM + XDIM) -> (1, BC, HDIM)
                x_gate = tanh_(x_gate); // -> hidden state update flow

                auto z_gate_projected = MiniTorch<BC * HIDDEN_DIM>::FlexibleMul(z_gate, -1);
                z_gate_projected = MiniTorch<BC * HIDDEN_DIM>::FlexibleAdd(z_gate_projected, 1); // == 1 - z_gate

                auto scaled_x_gate = MiniTorch<BC * HIDDEN_DIM>::template FlexibleMul<BC * HIDDEN_DIM>(x_gate, z_gate_projected); // == x_gate * (1 - z_gate)

                auto scaled_hidden = MiniTorch<BC * HIDDEN_DIM>::template FlexibleMul<BC * HIDDEN_DIM>(hidden, z_gate); // == hidden * z_gate

                auto final_hidden = MiniTorch<BC * HIDDEN_DIM>::template FlexibleAdd<BC * HIDDEN_DIM>(scaled_hidden, scaled_x_gate); // == (hidden * z_gate) + (x_gate * (1 - z_gate))

                hidden = final_hidden;
            }(), ...);
        };

        loop(std::make_index_sequence<L>{});

        hidden = MiniTorch<BC * HIDDEN_DIM>::TakeTranspose(hidden, 1, 0); // -> (1, BC, HDIM) -> (BC, 1, HDIM)

        return hidden;
    }

    SPTR<Tensor<HIDDEN_DIM * BC>> operator()(const SPTR<Tensor<T>> &x) { return this->forward(x); }
};
