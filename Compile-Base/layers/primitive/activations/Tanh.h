#pragma once

template<int T>
class Tanh {
public:
    explicit Tanh() = default;

    SPTR<Tensor<T>> forward(SPTR<Tensor<T>> x){
        auto t1 = MiniTorch<T>::Tanh(x);

        return t1;
    };

    SPTR<Tensor<T>> operator()(const SPTR<Tensor<T>> &x) { return forward(x); }
};
