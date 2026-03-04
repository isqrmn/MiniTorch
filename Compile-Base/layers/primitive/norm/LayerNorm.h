#pragma once

template<LLI T, LLI B, LLI C, int D>
class LayerNorm {
    SPTR<Tensor<D>> scale; // {D}
    SPTR<Tensor<D>> shift; // {D}

    void Initialization(){
        auto scale = MiniTorch<D>::HeUniformTensorInitialization(D); // {D}
        auto shift = MiniTorch<D>::HeUniformTensorInitialization(D); // {D}

        setScale(scale);
        setShift(shift);
    }

public:
    explicit LayerNorm(){ Initialization(); }

    SPTR<Tensor<D>> getScale(){ return this->scale; }

    SPTR<Tensor<D>> getShift(){ return this->shift; }

    void setScale(const SPTR<Tensor<D>> &scale){ this->scale = scale; }

    void setShift(const SPTR<Tensor<D>> &shift){ this->shift = shift; }

    SPTR<Tensor<T>> forward(const SPTR<Tensor<T>> &x){
        // X : {B, C, H, W}
        const SPTR<Element> &epsilon = std::make_shared<Element>(.0001);

        const auto &scale_ref = scale->getData();
        const auto &shift_ref = shift->getData();

        const auto &ref = x->getData();

        auto out = x->Copy();
        for (int i = 0; i < C * B; ++i) {
            auto x_c_mean = MiniTorch<T>::Mean(x, D, i * D);

            auto x_c_std = MiniTorch<T>::Std(x, D, i * D);
            x_c_std = MiniTorch<T>::AddElement(x_c_std, epsilon);

            for (int d = 0; d < D; ++d) {
                auto normalized = MiniTorch<T>::DivElement(MiniTorch<T>::SubElement(ref[i * D+ d], x_c_mean), x_c_std); // ((x - mean) / (Std + epsilon)) = alpha
                normalized = MiniTorch<T>::MulElement(normalized, scale_ref[d]); // alpha * scale
                normalized = MiniTorch<T>::AddElement(normalized, shift_ref[d]); // alpha * scale + shift

                out->SetObject(i * D + d, normalized);
            }
        }

        return out;
    }

    SPTR<Tensor<T>> operator()(const SPTR<Tensor<T>> &x) { return forward(x); }
};
