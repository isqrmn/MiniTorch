#pragma once

class RMSNorm {
    PTR_T scale; // {C}

    LLI D;

    void Initialization(){
        auto scale = Minitorch::HeUniformTensorInitialization(D, D); // {D}

        setScale(scale);
    }

public:
    explicit RMSNorm(LLI D) : D(D) {
        Initialization();
    }

    PTR_T getScale(){ return this->scale; }

    void setScale(const PTR_T &scale){ this->scale = scale; }

    PTR_T forward(const PTR_T &x){
        // X : {B, C, 1, D}
        const PTR_E &epsilon = std::make_shared<Element>(1e-4);

        const int B = x->getShape()[x->getShape().size() - 4];
        const int C = x->getShape()[x->getShape().size() - 3];

        const auto &scale_ref = scale->getData();

        const auto &ref = x->getData();

        auto out = x;
        for (int i = 0; i < C * B; ++i) {
            auto x_c_std = Minitorch::RMS(x, D, i * D);
            x_c_std = Minitorch::AddElement(x_c_std, epsilon);

            for (int d = 0; d < D; ++d) {
                auto normalized = Minitorch::DivElement(ref[i * D + d], x_c_std); // ((x - mean) / (Std + epsilon)) = alpha
                normalized = Minitorch::MulElement(normalized, scale_ref[d]); // alpha * scale

                out->SetObject(i * D + d, normalized);
            }
        }

        return out;
    }

    PTR_T operator()(const PTR_T &x) { return forward(x); }
};
