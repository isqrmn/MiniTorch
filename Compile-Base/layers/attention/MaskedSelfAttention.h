#pragma once

#include "../primitive/activations/Softmax.h"

template<LLI T, LLI BC, LLI L, LLI D>
requires (T == BC * L * D)
class MaskedSelfAttention {
    Softmax<BC * L * L, L> softmax_ = Softmax<BC * L * L, L>();

public:
    explicit MaskedSelfAttention() = default;

    SPTR<Tensor<T>> forward(const SPTR<Tensor<T>> &q, const SPTR<Tensor<T>> &k, const SPTR<Tensor<T>> &v) {
        // q - k - v : {B * nhead, L, D / nhead}

        // q @ k.transpose(-1, -2) => {B * nhead, L, L}
        // so(q @ k.transpose(-1, -2)) => {B * nhead, L, L}
        // so(q @ k.transpose(-1, -2)) @ v => {B * nhead, L, D / nhead}

        auto k_t = MiniTorch<T>::TakeTranspose(k, 2, 1); // {B * nhead, L, D / nhead} -> {B * nhead, D / nhead, L}

        auto matmul = MiniTorch<T>::template MatMul<T, BC * L * L>(q, k_t); // -> {BC, L, L}

        auto mask = MiniTorch<L>::Tril();
        mask = MiniTorch<L * L>::UnSqueeze(mask, 0);

        matmul = MiniTorch<BC * L * L>::template FlexibleMul<L * L>(matmul, mask);

        mask = MiniTorch<L * L>::Where(mask, 0, -9999, 0);

        matmul = MiniTorch<BC * L * L>::template FlexibleAdd<L * L>(matmul, mask);

        matmul = softmax_(matmul); // -> {BC, L, L}

        auto out = MiniTorch<BC * L * L>::template MatMul<T, T>(matmul, v); // -> {BC, L, D / nhead}

        return out;
    }

    SPTR<Tensor<T>> operator()(const SPTR<Tensor<T>> &q, const SPTR<Tensor<T>> &k, const SPTR<Tensor<T>> &v) { return forward(q, k, v); }
};