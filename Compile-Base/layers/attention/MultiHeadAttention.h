#pragma once

#include "../transformers/SelfAttention.h"
#include "../../CMiniTorch.h"
#include "../primitive/Linear.h"

template<LLI T, LLI nHead, LLI B, LLI L, LLI D>
requires (D % nHead == 0)
class MultiHeadAttention {
    Linear<T, B * L * nHead, D / nHead, D / nHead> k_proj = Linear<T, B * L * nHead, D / nHead, D / nHead>();
    Linear<T, B * L * nHead, D / nHead, D / nHead> q_proj = Linear<T, B * L * nHead, D / nHead, D / nHead>();
    Linear<T, B * L * nHead, D / nHead, D / nHead> v_proj = Linear<T, B * L * nHead, D / nHead, D / nHead>();
    Linear<T, B * L, D, D> out_proj = Linear<T, B * L, D, D>();

    SelfAttention<T, B * nHead, L, D / nHead> self_att = SelfAttention<T, B * nHead, L, D / nHead>();

public:
    explicit MultiHeadAttention() = default;

    SPTR<Tensor<T>> forward(const SPTR<Tensor<T>> &x) {
        // X : {B, 1, L, D}

        auto x_headed = MiniTorch<T>::ReShape(x, {B, L, nHead, D / nHead}); // -> {B, L, nHead, D / nhead}
        x_headed = MiniTorch<T>::TakeTranspose(x_headed, 2, 1); // -> {B, nhead, L, D / nhead}
        x_headed = MiniTorch<T>::ReShape(x, {B * nHead, L, D / nHead}); // -> {B * nhead, L, D / nhead}

        auto k = k_proj(x_headed);
        auto q = q_proj(x_headed);
        auto v = v_proj(x_headed);

        auto out = self_att(q, k, v);

        out = MiniTorch<T>::ReShape(out, {B, nHead, L, D / nHead});
        out = MiniTorch<T>::TakeTranspose(out, 2, 1); // -> {B, L, nhead, D / nhead}
        out = MiniTorch<T>::ReShape(out, {B, L, D}); // -> {B, L, D}

        out = out_proj(out); // -> {B, L, D}

        out = MiniTorch<T>::UnSqueeze(out, 1); // -> {B, 1, L, D}

        return out;
    }

    SPTR<Tensor<T>> operator()(const SPTR<Tensor<T>> &x) { return forward(x); }
};
