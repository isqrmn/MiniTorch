#pragma once

template<int T>
class Softplus {
public:
    explicit Softplus() = default;

    SPTR<Tensor<T>> forward(SPTR<Tensor<T>> x) {
        auto t1 = MiniTorch<T>::Softplus(x);

        return t1;
    };

    SPTR<Tensor<T>> operator()(const SPTR<Tensor<T>> &x) { return forward(x); }
};
