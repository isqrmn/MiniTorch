#pragma once

template<LLI T, LLI B, LLI C, LLI D>
class RMSNorm {
    SPTR<Tensor<D>> scale; // {C}

    void Initialization(){
        auto scale = MiniTorch<D>::HeUniformTensorInitialization(D); // {D}

        setScale(scale);
    }

public:
    explicit RMSNorm(){ Initialization(); }

    SPTR<Tensor<D>> getScale(){ return this->scale; }

    void setScale(const SPTR<Tensor<D>> &scale){ this->scale = scale; }

    SPTR<Tensor<T>> forward(const SPTR<Tensor<T>> &x){
        // X : {B, C, H, W}
        const SPTR<Element> &epsilon = std::make_shared<Element>(.0001);

        const auto &scale_ref = scale->getData();

        const auto &ref = x->getData();

        auto out = x;
        for (int i = 0; i < C * B; ++i) {
            auto x_c_std = MiniTorch<T>::RMS(x, D, i * D);
            x_c_std = MiniTorch<T>::AddElement(x_c_std, epsilon);

            for (int d = 0; d < D; ++d) {
                auto normalized = MiniTorch<T>::DivElement(ref[i * D+ d], x_c_std); // ((x - mean) / (Std + epsilon)) = alpha
                // normalized = MiniTorch<T>::MulElement(normalized, scale_ref[d]); // alpha * scale

                out->SetObject(i * D + d, normalized);
            }
        }

        return out;
    }

    SPTR<Tensor<T>> operator()(const SPTR<Tensor<T>> &x) { return forward(x); }
};
