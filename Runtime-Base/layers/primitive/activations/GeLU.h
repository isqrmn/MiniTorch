#pragma once

class GeLU {
public:
    explicit GeLU() = default;

    PTR_T forward(PTR_T x){
        auto t1 = Minitorch::GeLU(x);

        return t1;
    };

    PTR_T operator()(const PTR_T &x) { return forward(x); }
};
