#pragma once

template<int T>
class CrossEntropyLoss {
public:
    explicit CrossEntropyLoss() = default;

    SPTR<Tensor<T>> forward(const SPTR<Tensor<T>> &x, const SPTR<Tensor<T>> &y){
        SPTR<Tensor<T>> loss = MiniTorch<T>::CategoricalCrossEntropy(x, y);

        return loss;
    };

    SPTR<Tensor<T>> operator()(const SPTR<Tensor<T>> &x, const SPTR<Tensor<T>> &y) { return forward(x, y); }
};
