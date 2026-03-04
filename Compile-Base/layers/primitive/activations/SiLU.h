#pragma once

template<int T>
class SiLU {
public:
    explicit SiLU() = default;

    SPTR<Tensor<T>> forward(SPTR<Tensor<T>> x){
        auto t1 = MiniTorch<T>::SiLU(x);

        return t1;
    };

    SPTR<Tensor<T>> operator()(const SPTR<Tensor<T>> &x) { return forward(x); }
};
