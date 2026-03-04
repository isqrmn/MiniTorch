#pragma once

template<LLI T, LLI B, LLI C>
class BatchNorm {
    SPTR<Tensor<C>> scale; // {C}
    SPTR<Tensor<C>> shift; // {C}

    void Initialization(){
        auto scale = MiniTorch<C>::HeUniformTensorInitialization(C); // {C}
        auto shift = MiniTorch<C>::HeUniformTensorInitialization(C); // {C}

        setScale(scale);
        setShift(shift);
    }

public:
    explicit BatchNorm(){ Initialization(); }

    SPTR<Tensor<C>> getScale(){ return this->scale; }

    SPTR<Tensor<C>> getShift(){ return this->shift; };

    void setScale(const SPTR<Tensor<C>> &scale){ this->scale = scale; }

    void setShift(const SPTR<Tensor<C>> &shift){ this->shift = shift; }

    SPTR<Tensor<T>> forward(const SPTR<Tensor<T>> &x){
        // X : {B, C, H, W}
        const SPTR<Element> &epsilon = std::make_shared<Element>(.0001);

        const auto &scale_ref = scale->getData();
        const auto &shift_ref = shift->getData();

        auto x_t = MiniTorch<T>::TakeTranspose(x, 1, 0); // {B, C, H, W} -> {C, B, H, W}
        const auto &ref = x_t->getData();

        auto out = x_t;
        for (int i = 0; i < C; ++i) {
            auto x_c_mean = MiniTorch<T>::Mean(x_t, T / C, i * T / C);

            auto x_c_std = MiniTorch<T>::Std(x_t, T / C, i * T / C);
            x_c_std = MiniTorch<T>::AddElement(x_c_std, epsilon);

            for (int d = 0; d < T / C; ++d) {
                auto normalized = MiniTorch<T>::DivElement(MiniTorch<T>::SubElement(ref[i * T / C + d], x_c_mean), x_c_std); // ((x - mean) / (Std + epsilon)) = alpha
                normalized = MiniTorch<T>::MulElement(normalized, scale_ref[i]); // alpha * scale
                normalized = MiniTorch<T>::AddElement(normalized, shift_ref[i]); // alpha * scale + shift

                out->SetObject(i * T / C + d, normalized);
            }
        }
        out = MiniTorch<T>::TakeTranspose(out, 1, 0); // -> {C, B, H, W} -> {B, C, H, W}

        return out;
    }

    SPTR<Tensor<T>> operator()(const SPTR<Tensor<T>> &x) { return forward(x); }
};
