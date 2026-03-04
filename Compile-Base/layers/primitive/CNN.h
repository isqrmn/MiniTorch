#pragma once

#include <array>

// 3 4 5

template<int T, int C_IN, int C_OUT, int K1, int K2, bool Padding=true, int S1=1, int S2=1>
class CNN {
    ARR<SPTR<Tensor<C_IN * K1 * K2>>, C_OUT> weights;
    ARR<SPTR<Tensor<C_IN>>, C_OUT> bias_weights;

    void Initialization(){
        constexpr int one_filter_count = C_IN * K1 * K2;

        ARR<SPTR<Tensor<one_filter_count>>, C_OUT> total_weights;
        ARR<SPTR<Tensor<C_IN>>, C_OUT> total_bias;

        for (int i = 0; i < C_OUT; ++i) {
            SPTR<Tensor<one_filter_count>> weights = MiniTorch<one_filter_count>::HeUniformTensorInitialization(one_filter_count);
            SPTR<Tensor<C_IN>> bias = MiniTorch<C_IN>::HeUniformTensorInitialization(C_IN);

            const VEC_I temp_w{1, C_IN, K1, K2};
            weights = MiniTorch<one_filter_count>::ReShape(weights, temp_w);

            const VEC_I temp_b{1, C_IN, 1, 1};
            bias = MiniTorch<C_IN>::ReShape(bias, temp_b);

            total_weights[i] = weights;
            total_bias[i] = bias;
        }

        setBias(total_bias);
        setWeights(total_weights);
    };

public:
    explicit CNN(){Initialization();}

    static int getInFilters() { return C_IN; }

    static int getOutFilters() { return C_OUT; }

    static ARR<int, 2> getKernelSize() { return {K1, K2}; }

    static ARR<int, 2> getStrideSize() { return {S1, S2}; }

    static bool getPadding() { return Padding; };

    ARR<SPTR<Tensor<C_IN * K1 * K2>>, C_OUT> getWeights() const { return this->weights; }

    ARR<SPTR<Tensor<C_IN>>, C_OUT> getBias() const { return this->bias_weights; }

    void setWeights(const ARR<SPTR<Tensor<C_IN * K1 * K2>>, C_OUT> &weights) { this->weights = weights; }

    void setBias(const ARR<SPTR<Tensor<C_IN>>, C_OUT> &bias_weights) { this->bias_weights = bias_weights; }

    template<int B, int H, int W>
    SPTR<Tensor<B * C_OUT * H * W>> forward(const SPTR<Tensor<B * C_IN * H * W>>& x) const requires (Padding) {
        // slice - FlexMul - return
        // X : (B, C_OUT, H, W)-> (B, C_OUT, K1, K2)-Weights: (OUT, C_OUT, K1, K2)
        constexpr int one_filter_count = C_IN * K1 * K2;

        constexpr int P_h = K1 - 1;
        constexpr int P_w = K2 - 1;

        auto x_padded = MiniTorch<B * C_IN * H * W>::template PadInput<B, C_IN, H, W, P_h, P_w>(x);

        SPTR<Tensor<B * C_OUT * H * W>> out = std::make_shared<Tensor<B * C_OUT * H * W>>();
        out = MiniTorch<B * C_OUT * H * W>::ReShape(out, {C_OUT, H, W, B});

        int added = 0;
        for (int channel = 0; channel < C_OUT; channel++) {
            const SPTR<Tensor<one_filter_count>> &weight = getWeights()[channel];
            const SPTR<Tensor<C_IN>> &bias = getBias()[channel];

            for (int row = 0; row < x->getShape()[3]; row++) {
                for (int column = 0; column < x->getShape()[2]; column++) {
                    ARR<int, 4> x_start = {0, 0, row, column};
                    ARR<int, 4> x_end = {B - 1, C_IN - 1, row + K1 - 1, column + K2 - 1};

                    SPTR<Tensor<one_filter_count * B>> mul = MiniTorch<B * C_IN * (H + P_h) * (W + P_w)>::template FlexibleMul<B, one_filter_count, 4>(x_padded, weight, x_start, x_end); // (B, C_IN, K1, K2) * (1, C_IN, K1, K2) -> (B, C_IN, K1, K2)

                    SPTR<Tensor<B * C_IN * K2>> row_summed = MiniTorch<B * C_IN * K1 * K2>::template Sum<K1>(mul, 2);
                    SPTR<Tensor<B * C_IN>> column_summed = MiniTorch<B * C_IN * K2>::template Sum<K2>(row_summed, 3);

                    SPTR<Tensor<B * C_IN>> biased = MiniTorch<B * C_IN>::template FlexibleAdd<C_IN>(column_summed, bias);

                    SPTR<Tensor<B>> channel_summed = MiniTorch<B * C_IN>::template Sum<C_IN>(biased, 1);

                    for (int b = 0; b < B; ++b) {
                        auto result = MiniTorch<B>::TakeObject(channel_summed, b);

                        out->SetObject(added, result);

                        ++added;
                    }
                }
            }
        }

        out = MiniTorch<B * C_OUT * H * W>::TakeTranspose(out, 1, 3); // {C_OUT, H, W, B} -T> {C_OUT, B, W, H}
        out = MiniTorch<B * C_OUT * H * W>::TakeTranspose(out, 2, 3); // {C_OUT, B, W, H} -T> {C_OUT, B, H, W}
        out = MiniTorch<B * C_OUT * H * W>::TakeTranspose(out, 0, 1); // {C_OUT, B, H, W} -T> {B, C_OUT, H, W}

        return out;
    };

    SPTR<Tensor<T>> operator()(const SPTR<Tensor<T>> &x) const { return forward<1>(x); }
};
