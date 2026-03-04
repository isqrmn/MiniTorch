#pragma once

template<int T>
class MAE {
public:
    explicit MAE() = default;

    SPTR<Tensor<T>> forward(const SPTR<Tensor<T>> &x, const SPTR<Tensor<T>> &y){
        SPTR<Tensor<T>> loss = MiniTorch<T>::MAE(x, y);

        return loss;
    };

    SPTR<Tensor<T>> operator()(const SPTR<Tensor<T>> &x, const SPTR<Tensor<T>> &y) { return forward(x, y); }
};
