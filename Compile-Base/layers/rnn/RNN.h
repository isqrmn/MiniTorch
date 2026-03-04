#pragma once

#include "../primitive/Linear.h"
#include "../primitive/activations/Tanh.h"

template<int T, int BC, int F_IN, int HIDDEN_DIM>
class RNN {
    bool bias = true;

    Linear<BC * (F_IN + HIDDEN_DIM), BC, F_IN + HIDDEN_DIM, HIDDEN_DIM> x_linear = typename Linear<BC * (F_IN + HIDDEN_DIM), BC, F_IN + HIDDEN_DIM, HIDDEN_DIM>::Linear(bias);

    Tanh<BC * HIDDEN_DIM> tanh_ = typename Tanh<BC * HIDDEN_DIM>::Tanh();

public:
    explicit RNN(const bool bias) : bias(bias) {}

    explicit RNN() {}

    static int getFin() { return F_IN; }

    static int getHiddenDim() { return HIDDEN_DIM; }

    SPTR<Tensor<F_IN * HIDDEN_DIM>> getWeights() const{ return x_linear->getWeights(); }

    SPTR<Tensor<HIDDEN_DIM>> getBias() const{ return x_linear->getBias(); }

    void setXWeights(const SPTR<Tensor<(F_IN + HIDDEN_DIM) * HIDDEN_DIM>> &weights){ x_linear.setWeights(weights); }

    void setXBias(const SPTR<Tensor<HIDDEN_DIM>> &bias){ x_linear.setBias(bias); }

    SPTR<Tensor<HIDDEN_DIM * BC>> forward(SPTR<Tensor<T>> x){
        // X : (B, C, L, DIM) -> (BC, L, DIM)
        // Hidden : (B, C, 1, DIM) -> (BC, DIM)
        constexpr int L = T / BC / F_IN;

        SPTR<Tensor<BC * HIDDEN_DIM>> hidden = MiniTorch<BC * HIDDEN_DIM>::Zeros(); // -> (HDIM * BC)
        hidden = MiniTorch<BC * HIDDEN_DIM>::ReShape(hidden, {1, BC, HIDDEN_DIM});// -> (1, BC, HDIM)

        x = MiniTorch<T>::ReShape(x, {BC, L, F_IN}); // -> (BC, L, XDIM)
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

                auto x_gate = x_linear(catted_rz_input_flow); // -> (1, BC, HDIM + XDIM) -> (1, BC, HDIM)
                x_gate = tanh_(x_gate);

                auto final_hidden = MiniTorch<BC * HIDDEN_DIM>::template FlexibleAdd<BC * HIDDEN_DIM>(hidden, x_gate); // == (hidden * z_gate) + (x_gate * (1 - z_gate))

                hidden = final_hidden;
            }(), ...);
        };

        loop(std::make_index_sequence<L>{});

        hidden = MiniTorch<BC * HIDDEN_DIM>::TakeTranspose(hidden, 1, 0); // -> (1, BC, HDIM) -> (BC, 1, HDIM)

        return hidden;
    }

    SPTR<Tensor<HIDDEN_DIM * BC>> operator()(const SPTR<Tensor<T>> &x) { return this->forward(x); }

};
