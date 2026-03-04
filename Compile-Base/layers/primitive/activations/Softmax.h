#pragma once

template<LLI T, LLI D>
class Softmax {
public:
    explicit Softmax() = default;

    SPTR<Tensor<T>> forward(SPTR<Tensor<T>> x){
        auto t1 = MiniTorch<T>::template Softmax<D>(x);

        return t1;
    };

    SPTR<Tensor<T>> operator()(const SPTR<Tensor<T>> &x) { return forward(x); }
};
