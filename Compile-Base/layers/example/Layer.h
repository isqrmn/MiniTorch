#pragma once

#include "MultiHeadAttention.h"
#include "../primitive/Linear.h"
#include "../primitive/activations/GeLU.h"
#include "../primitive/norm/LayerNorm.h"

template<LLI T, LLI B, LLI L, LLI D>
requires (T == B * L * D)
class Layer {
    LayerNorm<T, B, L, D> layer_norm = LayerNorm<T, B, L, D>();
    MultiHeadAttention<T, 1, B, L, D> mha = MultiHeadAttention<T, 1, B, L, D>();

    class MLP {
        Linear<T, L, D, D> in_linear = Linear<T, L, D, D>();
        Linear<T, L, D, D> out_linear = Linear<T, L, D, D>();

        GeLU<T> gelu_ = GeLU<T>();
    public:
        explicit constexpr MLP() = default;

        SPTR<Tensor<T>> forward(const SPTR<Tensor<T>> &x) {
            auto x2 = in_linear(x);
            auto x3 = gelu_(x2);
            auto x4 = out_linear(x3);

            return x4;
        }

        SPTR<Tensor<T>> operator()(const SPTR<Tensor<T>> &x) { return forward(x); }
    };

    MLP mlp_ = MLP();

public:
    explicit Layer() = default;

    SPTR<Tensor<T>> forward(const SPTR<Tensor<T>> &x) {
        // X : {1, 1, L, D}
        auto x2 = layer_norm(x);
        auto x3 = mha(x2);
        auto x4 = MiniTorch<T>::template FlexibleAdd<T>(x3, x);

        auto x5 = layer_norm(x4);

        x5 = MiniTorch<T>::Squeeze(x5, 0);
        x4 = MiniTorch<T>::Squeeze(x4, 0); // {1, L, D} -> Linear icin

        auto x6 = mlp_(x5);
        auto x7 = MiniTorch<T>::template FlexibleAdd<T>(x6, x4);

        x7 = MiniTorch<T>::UnSqueeze(x7, 0);

        return x7;
    }

    SPTR<Tensor<T>> operator()(const SPTR<Tensor<T>> &x) { return forward(x); }
};
