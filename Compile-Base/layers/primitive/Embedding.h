#pragma once

template<int C, int D>
class Embedding {
    SPTR<Tensor<D * C>> weights;

    void Initialization() {
        // 1, C x C, D -> 1 D
        SPTR<Tensor<D * C>> weights = MiniTorch<D * C>::HeUniformTensorInitialization(C); // 1, D

        const VEC_I w_shape{1, C, D};
        weights = MiniTorch<D * C>::ReShape(weights, w_shape);

        setWeights(weights);
    };

public:
    explicit Embedding(){ Embedding<C, D>::Initialization();}

    static int getDim() { return D; }

    static int getCount() { return C; }

    SPTR<Tensor<D * C>>  getWeights() const {return this->weights; }

    void setWeights(SPTR<Tensor<D * C>> weights) { this->weights = weights; }

    SPTR<Tensor<D>> forward(const SPTR<Tensor<C>> &x) const {
        auto out = MiniTorch<C>::template MatMul<D * C, D>(x, weights);

        return out;
    };

    SPTR<Tensor<D>> forward(const int index) const{
        auto x = MiniTorch<C>::CreateOneHot(index);

        auto out = MiniTorch<C>::template MatMul<D * C, D>(x, weights);

        return out;
    };

    SPTR<Tensor<D>> operator()(const SPTR<Tensor<C>> &x) const { return this->forward(x); }

    SPTR<Tensor<D>> operator()(const int index) const {
        auto x = MiniTorch<C>::CreateOneHot(index);

        return operator()(x);
    }
};
