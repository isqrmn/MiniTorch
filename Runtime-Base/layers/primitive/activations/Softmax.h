#pragma once

class Softmax {
public:
    explicit Softmax() = default;

    PTR_T forward(PTR_T x){
        auto t1 = Minitorch::Softmax(x);

        return t1;
    };

    PTR_T operator()(const PTR_T &x) { return forward(x); }
};
