#pragma once

template<int T>
class GeLU {
public:
    explicit GeLU() = default;

    SPTR<Tensor<T>> forward(SPTR<Tensor<T>> x){
        auto t1 = MiniTorch<T>::GeLU(x);

        return t1;
    };

    SPTR<Tensor<T>> operator()(const SPTR<Tensor<T>> &x) { return forward(x); }
};
