#pragma once

class BatchNorm {
    PTR_T scale; // {C}
    PTR_T shift; // {C}

    LLI C;

    void Initialization(){
        auto scale = Minitorch::HeUniformTensorInitialization(C, C); // {C}
        auto shift = Minitorch::HeUniformTensorInitialization(C, C); // {C}

        setScale(scale);
        setShift(shift);
    }

public:
    explicit BatchNorm(LLI C) : C(C) {
        Initialization();
    }

    PTR_T getScale(){ return this->scale; }

    PTR_T getShift(){ return this->shift; };

    void setScale(const PTR_T &scale){ this->scale = scale; }

    void setShift(const PTR_T &shift){ this->shift = shift; }

    PTR_T forward(const PTR_T &x){
        // X : {B, C, H, W}
        const PTR_E &epsilon = std::make_shared<Element>(1e-4);

        const LLI T = x->getData().size();
        const LLI C = x->getShape()[x->getShape().size() - 3];

        const auto &scale_ref = scale->getData();
        const auto &shift_ref = shift->getData();

        auto x_t = Minitorch::TakeTranspose(x, 1, 0); // {B, C, H, W} -> {C, B, H, W}
        const auto &ref = x_t->getData();

        auto out = x_t;
        for (int i = 0; i < C; ++i) {
            auto x_c_mean = Minitorch::Mean(x_t, T / C, i * T / C);

            auto x_c_std = Minitorch::Std(x_t, T / C, i * T / C);
            x_c_std = Minitorch::AddElement(x_c_std, epsilon);

            for (int d = 0; d < T / C; ++d) {
                auto normalized = Minitorch::DivElement(Minitorch::SubElement(ref[i * T / C + d], x_c_mean), x_c_std); // ((x - mean) / (Std + epsilon)) = alpha
                // normalized = Minitorch::MulElement(normalized, scale_ref[i]); // alpha * scale
                // normalized = Minitorch::AddElement(normalized, shift_ref[i]); // alpha * scale + shift

                out->SetObject(i * T / C + d, normalized);
            }
        }
        out = Minitorch::TakeTranspose(out, 1, 0); // -> {C, B, H, W} -> {B, C, H, W}

        return out;
    }

    PTR_T operator()(const PTR_T &x) { return forward(x); }
};
