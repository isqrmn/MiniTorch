#pragma once

class SiLU {
public:
    explicit SiLU() = default;

    PTR_T forward(PTR_T x){
        auto t1 = Minitorch::SiLU(x);

        return t1;
    };

    PTR_T operator()(const PTR_T &x) { return forward(x); }
};
