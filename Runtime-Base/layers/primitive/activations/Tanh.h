#pragma once

class Tanh {
public:
    explicit Tanh() = default;

    PTR_T forward(PTR_T x){
        auto t1 = Minitorch::Tanh(x);

        return t1;
    };

    PTR_T operator()(const PTR_T &x) { return forward(x); }
};
