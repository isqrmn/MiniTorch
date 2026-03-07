#pragma once

class ReLU {
public:
    explicit ReLU() = default;

    PTR_T forward(PTR_T x){
        auto t1 = Minitorch::ReLU(x);

        return t1;
    };

    PTR_T operator()(const PTR_T &x) { return forward(x); }
};
