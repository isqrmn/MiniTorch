#pragma once

template<int T>
class MSE {
public:
    explicit MSE() = default;

    SPTR<Tensor<T>> forward(const SPTR<Tensor<T>> &x, const SPTR<Tensor<T>> &y){
        SPTR<Tensor<T>> loss = MiniTorch<T>::MSE(x, y);

        return loss;
    };

    SPTR<Tensor<T>> operator()(const SPTR<Tensor<T>> &x, const SPTR<Tensor<T>> &y) { return forward(x, y); }
};
