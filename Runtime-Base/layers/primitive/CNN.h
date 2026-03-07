#pragma once

// 3 4 5
class CNN {
    LLI C_IN;
    LLI C_OUT;

    LLI K1;
    LLI K2;

    LLI S1;
    LLI S2;

    bool Padding;

    VEC_T weights;
    VEC_T bias_weights;

    void Initialization(){
        const int one_filter_count = C_IN * K1 * K2;

        VEC_T total_weights;
        VEC_T total_bias;

        for (int i = 0; i < C_OUT; ++i) {
            PTR_T weights = Minitorch::HeUniformTensorInitialization(one_filter_count, one_filter_count);
            PTR_T bias = Minitorch::HeUniformTensorInitialization(C_IN, one_filter_count);

            const VEC_I temp_w{1, C_IN, K1, K2};
            weights = Minitorch::ReShape(weights, temp_w);

            const VEC_I temp_b{1, C_IN, 1, 1};
            bias = Minitorch::ReShape(bias, temp_b);

            total_weights.push_back(weights);
            total_bias.push_back(bias);
        }

        setBias(total_bias);
        setWeights(total_weights);
    }

public:
    explicit CNN(const LLI& C_IN = 1, const LLI& C_OUT = 1, const LLI& K1 = 2, const LLI& K2 = 2, const LLI& S1 = 1, const LLI& S2 = 1, const bool Padding=true) :
    C_IN(C_IN), C_OUT(C_OUT), K1(K1), K2(K2), S1(S1), S2(S2), Padding(Padding) {
        Initialization();
    }

    int getInFilters() const { return C_IN; }

    int getOutFilters() const { return C_OUT; }

    VEC_I getKernelSize() const { return {K1, K2}; }

    VEC_I getStrideSize() const { return {S1, S2}; }

    bool getPadding() const { return Padding; };

    VEC_T getWeights() const { return weights; }

    VEC_T getBias() const { return bias_weights; }

    void setWeights(const VEC_T &weights) { this->weights = weights; }

    void setBias(const VEC_T &bias_weights) { this->bias_weights = bias_weights; }

    PTR_T forward(const PTR_T& x) const {
        // slice - FlexMul - return
        // X : (B, C_OUT, H, W)-> (B, C_OUT, K1, K2)-Weights: (OUT, C_OUT, K1, K2)
        const int one_filter_count = C_IN * K1 * K2;

        const int xs_size = x->getShape().size();
        const int H = x->getShape()[xs_size - 2];
        const int W = x->getShape()[xs_size - 1];
        const int B = x->getShape()[0];

        const int P_h = K1 - 1;
        const int P_w = K2 - 1;

        auto x_padded = Minitorch::PadInput(x, P_w, P_h);

        x_padded->PrintTensor();

        PTR_T out = Minitorch::Zeros(C_OUT * B * H * W);
        out = Minitorch::ReShape(out, {C_OUT, H, W, B});

        int added = 0;
        for (int channel = 0; channel < C_OUT; channel++) {
            const PTR_T &weight = weights[channel];
            const PTR_T &bias = bias_weights[channel];

            for (int row = 0; row < x->getShape()[x->getShape().size() - 2] / S1; row++) {
                for (int column = 0; column < x->getShape()[x->getShape().size() - 1] / S2; column++) {
                    VEC_I x_start = {0, 0, row * S1, column * S2};
                    VEC_I x_end = {B - 1, C_IN - 1, row * S1 + K1 - 1, column * S2 + K2 - 1};

                    if (row == 2 && column == 0) {
                        int lpl = 0;
                    }

                    PTR_T mul = Minitorch::FlexibleMul(x_padded, weight, x_start, x_end); // (B, C_IN, K1, K2) * (1, C_IN, K1, K2) -> (B, C_IN, K1, K2)

                    PTR_T row_summed = Minitorch::Sum(mul, 2);
                    PTR_T column_summed = Minitorch::Sum(row_summed, 3);

                    PTR_T biased = Minitorch::FlexibleAdd(column_summed, bias);

                    PTR_T channel_summed = Minitorch::Sum(biased, 1);

                    for (int b = 0; b < B; ++b) {
                        auto result = Minitorch::TakeObject(channel_summed, b);

                        out->SetObject(added, result);

                        ++added;
                    }
                }
            }
        }

        out = Minitorch::TakeTranspose(out, 1, 3); // {C_OUT, H, W, B} -T> {C_OUT, B, W, H}
        out = Minitorch::TakeTranspose(out, 2, 3); // {C_OUT, B, W, H} -T> {C_OUT, B, H, W}
        out = Minitorch::TakeTranspose(out, 0, 1); // {C_OUT, B, H, W} -T> {B, C_OUT, H, W}

        return out;
    };

    PTR_T operator()(const PTR_T &x) const { return forward(x); }
};
