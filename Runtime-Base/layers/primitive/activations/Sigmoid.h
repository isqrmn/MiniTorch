#pragma once

class Sigmoid {
public:
    explicit Sigmoid() = default;

    PTR_T forward(PTR_T x){
        auto t1 = Minitorch::Sigmoid(x);

        return t1;
    };

    PTR_T operator()(const PTR_T &x) { return forward(x); }
};
