#pragma once

template<int T, int R, int F_IN, int F_OUT>
requires (T == (R * F_IN))
class Linear {
    const bool bias = true;

    SPTR<Tensor<F_IN * F_OUT>> weights;
    SPTR<Tensor<F_OUT>> bias_weight;

    void Initialization() {
        SPTR<Tensor<F_IN * F_OUT>> weights = MiniTorch<F_IN * F_OUT>::HeUniformTensorInitialization(F_IN);

        const VEC_I w_shape{1, F_IN, F_OUT};
        weights = MiniTorch<F_IN * F_OUT>::ReShape(weights, w_shape);

        setWeights(weights);

        if (bias) {
            SPTR<Tensor<F_OUT>> bias = MiniTorch<F_OUT>::HeUniformTensorInitialization(F_IN);

            const VEC_I b_shape{1, 1, F_OUT};
            bias = MiniTorch<F_OUT>::ReShape(bias, b_shape);

            setBias(bias);
        }
    }

public:
    explicit Linear(const bool bias): bias(bias) { Initialization();}

    explicit Linear() { Initialization();}

    SPTR<Tensor<F_IN * F_OUT>> getWeights(){ return weights; }

    SPTR<Tensor<F_OUT>> getBias(){ return bias_weight; }

    void setWeights(const SPTR<Tensor<F_IN * F_OUT>> &weights){ this->weights = weights; }

    void setBias(const SPTR<Tensor<F_OUT>> &bias){ this->bias_weight = bias; }

    SPTR<Tensor<R * F_OUT>> forward(const SPTR<Tensor<T>> &x) const{
        SPTR<Tensor<R * F_OUT>> out = MiniTorch<T>::template MatMul<F_IN * F_OUT, R * F_OUT>(x, weights);
        if (bias) {
            out = MiniTorch<R * F_OUT>::template FlexibleAdd<F_OUT>(out, bias_weight);
        }

        return out;
    }

    SPTR<Tensor<R * F_OUT>> operator()(const SPTR<Tensor<T>> &x) const { return forward(x); }
};
