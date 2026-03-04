#pragma once

template<int T>
class ReLU {
public:
    explicit ReLU() = default;

    SPTR<Tensor<T>> forward(SPTR<Tensor<T>> x){
        auto t1 = MiniTorch<T>::ReLU(x);

        return t1;
    };

    SPTR<Tensor<T>> operator()(const SPTR<Tensor<T>> &x) { return forward(x); }
};
