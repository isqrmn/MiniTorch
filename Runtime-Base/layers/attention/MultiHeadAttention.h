#pragma once

#include "../primitive/Linear.h"
#include "SelfAttention.h"

class MultiHeadAttention {
    LLI D;
    LLI N_HEAD;

    Linear out_proj;

    Linear k_proj;
    Linear q_proj;
    Linear v_proj;

    SelfAttention self_att = SelfAttention();

public:
    explicit MultiHeadAttention(LLI D, LLI N_HEAD=1) : D(D), N_HEAD(N_HEAD), out_proj(Linear(D, D)),
        k_proj(Linear(D / N_HEAD, D / N_HEAD)), q_proj(Linear(D / N_HEAD, D / N_HEAD)), v_proj(Linear(D / N_HEAD, D / N_HEAD)) {
        if (D % N_HEAD != 0) {
            throw std::runtime_error("Bad D % N_HEAD : It Should Be Zero!");
        }
    }

    PTR_T forward(const PTR_T &x) {
        // X : {B, 1, L, D}
        const int B = x->getShape()[0];
        const int L = x->getShape()[2];

        auto x_headed = Minitorch::ReShape(x, {B, L, N_HEAD, D / N_HEAD}); // -> {B, L, N_HEAD, D / N_HEAD}
        x_headed = Minitorch::TakeTranspose(x_headed, 2, 1); // -> {B, N_HEAD, L, D / N_HEAD}
        x_headed = Minitorch::ReShape(x, {1, B * N_HEAD, L, D / N_HEAD}); // -> {1, B * N_HEAD, L, D / N_HEAD}

        auto k = k_proj(x_headed);
        auto q = q_proj(x_headed);
        auto v = v_proj(x_headed); // {1, B * N_HEAD, L, D / N_HEAD} -> {1, B * N_HEAD, L, D / N_HEAD}

        auto out = self_att(q, k, v);

        out = Minitorch::ReShape(out, {B, N_HEAD, L, D / N_HEAD});
        out = Minitorch::TakeTranspose(out, 2, 1); // -> {B, L, N_HEAD, D / N_HEAD}
        out = Minitorch::ReShape(out, {1, B, L, D}); // -> {1, B, L, D}

        out = out_proj(out); // -> {1, B, L, D}

        out = Minitorch::ReShape(out, {B, 1, L, D}); // {1, B, L, D} -> {B, 1, L, D}

        return out;
    }

    PTR_T operator()(const PTR_T &x) { return forward(x); }
};
