#pragma once

#include "../primitive/activations/Softmax.h"

class SelfAttention {
    Softmax softmax_ = Softmax();

public:
    explicit SelfAttention() = default;

    PTR_T forward(const PTR_T &q, const PTR_T &k, const PTR_T &v) {
        // q - k - v : {1, B * nhead, L, D / nhead}

        // q @ k.transpose(-1, -2) => {1, B * nhead, L, L}
        // so(q @ k.transpose(-1, -2)) => {1, B * nhead, L, L}
        // so(q @ k.transpose(-1, -2)) @ v => {1, B * nhead, L, D / nhead}

        auto k_t = Minitorch::TakeTranspose(k, 3, 2); // {1, B * nhead, L, D / nhead} -> {1, B * nhead, D / nhead, L}

        auto matmul = Minitorch::MatMul(q, k_t); // -> {1, BC, L, L}

        matmul = softmax_(matmul); // -> {1, BC, L, L}

        auto out = Minitorch::MatMul(matmul, v); // -> {1, BC, L, D / nhead}

        return out;
    }

    PTR_T operator()(const PTR_T &q, const PTR_T &k, const PTR_T &v) { return forward(q, k, v); }
};
