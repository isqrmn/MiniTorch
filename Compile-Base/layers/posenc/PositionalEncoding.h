#pragma once

/*
 *
* class SCPositionalEncoding(nn.Module):
    def __init__(self, max_len: int = 64, ndim : int = 1536):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, ndim, 2) * (-torch.log(torch.tensor(10000.0)) / ndim))

        pe = torch.zeros(max_len, 1, ndim)

        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(2)]
 *
 */

#include <math.h>

template<LLI T, LLI D>
requires (D % 2 == 0)
class PositionalEncoding {
public:
    explicit PositionalEncoding() = default;

    SPTR<Tensor<T * D>> forward() {
        ARR<PTR_E, T * D> out;

        SPTR<Tensor<D / 2>> div_term = MiniTorch<D>::template Arange<0, 2>();
        div_term = MiniTorch<D / 2>::FlexibleMul(div_term, -(std::log(10000) / D));
        div_term = MiniTorch<D / 2>::Exp(div_term);

        const auto &d_ref = div_term->getData();
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < D; ++i) {
                if (i % 2 == 0){
                    out[i + t * D] = std::make_shared<Element>(std::sin(t * d_ref[i / 2]->getData()));
                }
                else {
                    out[i + t * D] = std::make_shared<Element>(std::cos(t * d_ref[(i - 1) / 2]->getData()));
                }
            }
        }

        auto t1 = std::make_shared<Tensor<T * D>>(out);

        t1 = MiniTorch<T * D>::ReShape(t1, {T, D});

        return t1;
    }

    SPTR<Tensor<T * D>> operator()() { return forward(); }
};
