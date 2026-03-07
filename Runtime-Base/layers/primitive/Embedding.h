#pragma once

class Embedding {
    PTR_T weights;

    LLI C;
    LLI D;

    void Initialization() {
        // 1, C x C, D -> 1 D
        PTR_T weights = Minitorch::HeUniformTensorInitialization(C * D, D); // 1, D

        const VEC_I w_shape{C, D};
        weights = Minitorch::ReShape(weights, w_shape);

        setWeights(weights);
    };

public:
    explicit Embedding(LLI C, LLI D): C(C), D(D) {
        Initialization();
    }

    explicit Embedding() = delete;

    int getDim() { return D; }

    int getCount() { return C; }

    PTR_T  getWeights() const {return this->weights; }

    void setWeights(PTR_T weights) { this->weights = weights; }

    PTR_T forward(const PTR_T &x) const {
        auto out = Minitorch::MatMul(x, weights);

        return out;
    }

    PTR_T forward(const int index) const{
        auto x = Minitorch::CreateOneHot(C, index);

        auto out = Minitorch::MatMul(x, weights);

        return out;
    }

    PTR_T operator()(const PTR_T &x) const { return this->forward(x); }

    PTR_T operator()(const int index) const {
        const auto x = Minitorch::CreateOneHot(C, index);

        return operator()(x);
    }
};
