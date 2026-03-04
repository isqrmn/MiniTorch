#pragma once

#include <cmath>
#include <random>
#include <fstream>
#include <iostream>
#include <variant>
#include "CTensor.h"

template<LLI T>
class MiniTorch {
    // %% INNER CONSEXPR %%
    template<LLI T2>
    static constexpr ARR<int, T2> CalcSliceShape(const ARR<int, T2> &start, const ARR<int, T2> &end) {
        ARR<int, T2> shape{};

        for (std::size_t i = 0; i < T2; ++i)
            shape[i] = end[i] - start[i] + 1;

        return shape;
    }

    // %% UTILITY FUNCS %%
    static std::vector<String> SplitByChar(const String &data, const char special){
        std::vector<String> out;

        int s = 0;
        for (int i = 0; i < data.size(); i++) {
            if (data[i] == special) {
                out.push_back(data.substr(s, i - s));
                s = i + 1;
            }
        }

        return out;
    }

    template<LLI A>
    static VEC_I CreateIndex(const SPTR<Tensor<A>> &t1, const int n) {
        if (n > t1->getData().size()) {
            throw std::runtime_error("Bad Index!");
        }

        const VEC_I &ref = t1->getIndexWeights();

        VEC_I index;
        index.reserve(t1->getShape().size());

        int temp_total = n;
        for (const int w1 : ref) {
            int temp_index = 0;

            while (w1 <= temp_total) {
                temp_index++;
                temp_total -= w1;
            }

            index.push_back(temp_index);
        }

        return index;
    }

    static VEC_I NonZero(const VEC_I &vec) {
        for (const int i : vec) {
            if (i <= 0) {
                throw std::runtime_error("Bad Shape!");
            }
        }

        return vec;
    }

    template<LLI A>
    static VEC_I TransformShape(const SPTR<Tensor<A>> &t1, const VEC_I &shape) {
        const int &size = t1->getData().size();

        VEC_I new_shape = shape;

        bool dynamic = false;

        int count = 1;
        for (const int i : shape) {
            if (i != -1) {
                count *= i;
            }else {
                dynamic = true;
            }
        }

        if (count != size && !dynamic) {
            throw std::runtime_error("Bad Shape!");
        }

        if (size % count != 0 && dynamic) {
            throw std::runtime_error("Bad Shape!");
        }

        if (dynamic) {
            for (int i = 0; i < shape.size(); ++i) {
                if (shape[i] == -1) {
                    new_shape[i] = size / count;
                }
            }
        }

        return new_shape;
    }

    template<LLI A>
    static VEC_I TransformIndex(const SPTR<Tensor<A>> &t1, const VEC_I &index) {
        const VEC_I &ref = t1->getShape();
        VEC_I transformed = index;

        if (transformed.size() != ref.size()) {
            throw std::runtime_error("Bad Index!");
        }

        for (int i = 0; i < index.size(); ++i) {
            if (index[i] < 0) {
                transformed[i] += ref[i];
            }
            if (transformed[i] < 0 || transformed[i] >= ref[i]) {
                throw std::runtime_error("Bad Index!");
            }
        }

        return transformed;
    }

    template<LLI B>
    static ARR<int, B> NonZero(const ARR<int, B> &vec) {
        for (const int i : vec) {
            if (i <= 0) {
                throw std::runtime_error("Bad Shape!");
            }
        }

        return vec;
    }

    template<LLI A, LLI B>
    static ARR<int, B> TransformShape(const SPTR<Tensor<A>> &t1, const ARR<int, B> &shape) {
        const int &size = t1->getData().size();

        VEC_I new_shape = shape;

        bool dynamic = false;

        int count = 1;
        for (const int i : shape) {
            if (i != -1) {
                count *= i;
            }else {
                dynamic = true;
            }
        }

        if (count != size && !dynamic) {
            throw std::runtime_error("Bad Shape!");
        }

        if (size % count != 0 && dynamic) {
            throw std::runtime_error("Bad Shape!");
        }

        if (dynamic) {
            for (int i = 0; i < shape.size(); ++i) {
                if (shape[i] == -1) {
                    new_shape[i] = size / count;
                }
            }
        }

        return new_shape;
    }

    template<LLI A, LLI B>
    static ARR<int, B> TransformIndex(const SPTR<Tensor<A>> &t1, const ARR<int, B> &index) {
        const VEC_I &ref = t1->getShape();
        ARR<int, B> transformed = index;

        if (transformed.size() != ref.size()) {
            throw std::runtime_error("Bad Index!");
        }

        for (int i = 0; i < index.size(); ++i) {
            if (index[i] < 0) {
                transformed[i] += ref[i];
            }
            if (transformed[i] < 0 || transformed[i] >= ref[i]) {
                throw std::runtime_error("Bad Index!");
            }
        }

        return transformed;
    }
    template<LLI A>
    static int TransformIndex(const SPTR<Tensor<A>> &t1, const int &index) {
        const int &size = t1->getData().size();
        int transformed = index;

        if (transformed < 0) {
            transformed += size;
        }

        if (transformed < 0 || transformed >= size) {
            throw std::runtime_error("Bad Index!");
        }

        return transformed;
    }

public:
    ~MiniTorch() = delete;
    MiniTorch() = delete;
    MiniTorch(const MiniTorch&) = delete;
    MiniTorch(MiniTorch&&) = delete;
    MiniTorch& operator=(const MiniTorch&) = delete;
    MiniTorch& operator=(MiniTorch&&) = delete;

    // %% SET TENSORS %%
    static void ZeroStartData(const SPTR<Tensor<T>> &t1) {
        for (int i = 0; i < T; ++i) {
            auto temp_e = std::make_shared<Element>(.0);
            t1->SetObject(i, temp_e);
        }
    }

    template<LLI T2>
    static void PartialStartData(const SPTR<Tensor<T>> &t1, const SPTR<Tensor<T2>> &t2) {
        const ARR<PTR_E, T> &ref = t1->getData();

        for (int i = 0; i < T; ++i) {
            t2->SetObject(i, ref[i]);
        }
    }

    template<LLI T2>
    static void PartialIndexStartData(const SPTR<Tensor<T>> &t1, const SPTR<Tensor<T2>> &t2) {
        const ARR<PTR_E, T> &ref = t1->getData();

        for (int i = 0; i < T; ++i) {
            auto ref_i = MiniTorch<T>::CreateIndex(t1, i);
            t2->SetObject(ref_i, ref[i]);
        }
    }

    // %% SOME CONSEXPR %%
    template<LLI T2>
    static constexpr int CalcSliceSize(const ARR<int, T2> &start, const ARR<int, T2> &end)
    {
        int w = 1;
        for (int i = 0; i < T2; ++i) {
            w *= end[i] - start[i] + 1;
        }

        return w;
    }

    // -> T3
    template<LLI MulCount, LLI T_RowSize, LLI T2_ColumnSize>
    static constexpr int CalcMatMulSize()
    {
        constexpr int C = ((T / MulCount) / T_RowSize);

        return C * T_RowSize * T2_ColumnSize;
    }

    // // %% TENSOR %%
    template<LLI T2, LLI w, std::array<int, T2> start_indexes, std::array<int, T2> end_indexes>
    static std::shared_ptr<CTensor<w>> TakeSlice(const std::shared_ptr<CTensor<T>> &t1) {
        // start_indexes = TransformIndex<T, T2>(t1, start_indexes);
        // end_indexes = TransformIndex<T, T2>(t1, end_indexes);

        const ARR<PTR_E, T> &ref = t1->getData();

        constexpr ARR<int, T2> n_shape = CalcSliceShape<T2>(start_indexes, end_indexes);

        ARR<PTR_E, w> elements;

        int add_index = 0;
        for (int n = 0; n < ref.size(); n++) {
            VEC_I cor_indexing = CreateIndex<T>(t1, n);

            bool add = true;
            for (int t = 0; t < start_indexes.size(); t++) {
                if ((cor_indexing[t] > end_indexes[t]) || (cor_indexing[t] < start_indexes[t])) {
                    add = false;
                    break;
                }
            }
            if (add) {
                elements[add_index] = ref[n]->Copy();
                ++add_index;
            }
        }

        VEC_I vec(n_shape.begin(), n_shape.end());

        return std::make_shared<Tensor<w>>(elements, vec);
    }

    template<LLI A>
    static PTR_E TakeObject(const SPTR<Tensor<A>> &t1, VEC_I indexing) {
        indexing = TransformIndex<A>(t1, indexing);

        const ARR<PTR_E, A> &ref = t1->getData();
        const VEC_I &ref2 = t1->getIndexWeights();

        int index = 0;
        for (int i=0; i<indexing.size(); i++) {
            index += indexing[i] * ref2[i];
        }

        return ref[index];
    }
    template<LLI A>
    static PTR_E TakeObject(const SPTR<Tensor<A>> &t1, int indexing) {
        indexing = TransformIndex<A>(t1, indexing);

        const ARR<PTR_E, A> &ref = t1->getData();

        return ref[indexing];
    }

    static SPTR<Tensor<T>> ReShape(const SPTR<Tensor<T>> &t1, const VEC_I &shape) {

        const VEC_I &new_shape = TransformShape<T>(t1, shape);

        const SPTR<Tensor<T>> &temp = t1->Copy();

        temp->setShape(new_shape);

        return temp;
    }

    template<LLI T2>
    static SPTR<Tensor<T2>> Repeat(const SPTR<Tensor<T>> &t1, const VEC_I &repeats) {
        NonZero(repeats);

        if (t1->getShape().size() != repeats.size()) {
            throw std::runtime_error("Bad Repeat!");
        }

        int cardinality = 0;
        for (const int i : repeats) {
            if (i > 1) {
                ++cardinality;
            }
        }

        if (cardinality > 2) {
            throw std::runtime_error("Bad Cardinality!");
        }

        const VEC_I &init_shape = t1->getShape();

        VEC_I new_shape;
        new_shape.reserve(repeats.size());
        for (int i = 0; i < repeats.size(); ++i) {
            new_shape.push_back(repeats[i] * t1->getShape()[i]);
        }

        // 2 3 4 - 2 3 5 -> 4 3 4 - 1 3 5 -T> 4 3 4
        // 4 3 4 - 1 3 5 -T> 3 4 4 - 3 1 5 -T> 4 9 4
        // 4 9 4 - 1 1 5 -T> 4 9 4 - 5 1 1 -T> 4 9 20
        // 4 9 20 - 1 1 1

        SPTR<Tensor<T2>> out = std::make_shared<Tensor<T2>>(new_shape); // s: 2 6 8 = 2 3 4 - r2 2 2
        MiniTorch<T2>::ZeroStartData(out);
        MiniTorch<T>::template PartialIndexStartData<T2>(t1, out);

        int last_cumr = 0;
        int cumr = 1;
        for (int rep_dim = repeats.size() - 1; rep_dim >= 0; --rep_dim) {
            for (int r = 1; r < repeats[rep_dim]; ++r) {
                for (int i = 0; i < T; ++i) {
                    auto temp_e = t1->TakeObject(i % T);
                    auto t1_index = MiniTorch<T>::CreateIndex<T>(t1, i % T); // 2r {0, 0, 0} -> {0, 0, 2} == {0, 0, 1} -> {0, 0, 3} == {0, 1, 0} -> {0, 1, 2}
                    // 3r {0, 0, 0} -> {0, 0, 2} -> {0, 0, 4} == {0, 0, 1} -> {0, 0, 3} -> {0, 0, 5}

                    t1_index[rep_dim] += init_shape[rep_dim] * r;
                    t1_index[rep_dim + last_cumr] -= init_shape[rep_dim + last_cumr];
                    for (int cr = 0; cr < cumr; ++cr) {
                        t1_index[rep_dim + last_cumr] += init_shape[rep_dim + last_cumr];
                        out->SetObject(t1_index, temp_e);
                    }
                }
            }
            cumr *= repeats[rep_dim];

            if (repeats[rep_dim] != 1) {
                last_cumr = 1;
            }else { ++last_cumr; }
        }

        return out;
    }

    static SPTR<Tensor<T>> Clamp(const SPTR<Tensor<T>> &t1, DTYPE min, DTYPE max) {
        const ARR<PTR_E, T> &ref = t1->getData();

        ARR<PTR_E, T> out;
        for (int i = 0; i < ref.size(); ++i) {
            if (ref[i]->getData() > max) {
                out[i] = std::make_shared<Element>(max, .0, .0);
            }
            else if (ref[i]->getData() < min) {
                out[i] = std::make_shared<Element>(min, .0, .0);
            }
            else {
                out[i] = std::make_shared<Element>(ref[i]->getData(), .0, ref[i]->getBackScalar0());
            }
        }

        return std::make_shared<Tensor<T>>(out, t1->getShape());
    }

    static SPTR<Tensor<T>> TakeTranspose(const SPTR<Tensor<T>> &t1, const int dim_a, const int dim_b) {
        const ARR<PTR_E, T> &ref = t1->getData();

        SPTR<Tensor<T>> out = t1->Copy();

        VEC_I n_shape = out->getShape();

        int temp = n_shape[dim_a];
        n_shape[dim_a] = n_shape[dim_b];
        n_shape[dim_b] = temp;

        out = ReShape(out, n_shape);

        for (int i = 0; i < ref.size(); i++) {
            VEC_I cor_indexing = CreateIndex<T>(t1, i);

            temp = cor_indexing[dim_a];
            cor_indexing[dim_a] = cor_indexing[dim_b];
            cor_indexing[dim_b] = temp;

            out->SetObject(cor_indexing, ref[i]->Copy());
        }

        return out;
    }

    static SPTR<Tensor<T>> UnSqueeze(const SPTR<Tensor<T>> &t1, const int dim) {
        VEC_I good_shape;

        for (int i=0; i <= t1->getShape().size(); i++) {
            if (i == dim && dim != t1->getShape().size()) {
                good_shape.push_back(1);
                good_shape.push_back(t1->getShape()[i]);
            }
            else if (i == dim){
                good_shape.push_back(1);
            }
            else if (i < t1->getShape().size()) {
                good_shape.push_back(t1->getShape()[i]);
            }
        }

        SPTR<Tensor<T>> out_tensor = t1->Copy();
        out_tensor = ReShape(out_tensor, good_shape);

        return out_tensor;
    }

    static SPTR<Tensor<T>> Squeeze(const SPTR<Tensor<T>> &t1, const int dim) {
        VEC_I good_shape;

        for (int i=0; i < t1->getShape().size(); i++) {
            if (i == dim && t1->getShape()[i] < 2) {
                continue;
            }
            good_shape.push_back(t1->getShape()[i]);
        }

        const SPTR<Tensor<T>> &temp = t1->Copy();

        temp->setShape(good_shape);

        return temp;
    }

    static SPTR<Tensor<T>> ViewAsComplex(const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        if (t1->complex) {
            return t1->Copy();
        }

        VEC_I &shape = t1->getShape();
        shape[shape.size()] /= 2; // (3,4,10) -> (3,4,5)
        shape.push_back(2); // (3,4,5) -> (3,4,5,2)

        ARR<PTR_E, T> out;
        for (int i = 0; i < t1->data.size(); i ++) {
            PTR_E temp = std::make_shared<Element>(ref[i]->getData());
            out.push_back(temp);

            PTR_E tempim = std::make_shared<Element>(ref[i+1]->getImData());
            out.push_back(tempim);
        }

        return std::make_shared<Tensor>(out, shape, true);
    }

    static SPTR<Tensor<T>> ViewAsReal(const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        if (!t1->getComplex()) {
            return t1->Copy();
        }

        VEC_I shape = t1->getShape();
        shape.pop_back(); // (3,4,5,2) -> (3,4,5)
        shape[shape.size()] *= 2; // (3,4,5) -> (3,4,10)

        ARR<PTR_E, T> out;
        for (int i = 0; i < t1->getData().size(); ++++i) {
            PTR_E temp = std::make_shared<Element>(ref[i]->getData(), .0);
            PTR_E temp2 = std::make_shared<Element>(ref[i]->getImData(), .0);
            out[i] = temp;
            out[i + 1] = temp2;
        }

        return std::make_shared<Tensor<T>>(out, shape, false);
    }

    static ARR<DTYPE, T> AsVector(const SPTR<Tensor<T>> &t1){
        const ARR<PTR_E, T> &ref = t1->getData();

        ARR<DTYPE, T> out;
        for (int i = 0; i < ref.size(); ++i) {
            out[i] = ref[i]->getData();
        }

        return out;
    }

    static PTR_E Amax(const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        PTR_E amax = t1->getData()[0]->Copy();
        for (int i = 1; i < ref.size(); i++) {
            if (ref[i]->getData() > amax->getData()) {
                amax = ref[i];
            }
        }

        return amax;
    }

    static PTR_E Amin(const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        PTR_E amin = ref[0]->Copy();
        for (int i = 1; i < ref.size(); i++) {
            if (ref[i]->getData() < amin->getData()) {
                amin = ref[i];
            }
        }

        return amin;
    }

    template<LLI T2>
    static SPTR<Tensor<T / T2>> Sum(const SPTR<Tensor<T>> &t1, int dim) {
        SPTR<Tensor<T>> temp_t1;
        if (dim != 0) {
            temp_t1 = MiniTorch<T>::TakeTranspose(t1, dim, 0);
        }
        else {
            temp_t1 = t1;
        }

        VEC_I new_shape = temp_t1->getShape();
        new_shape[0] = 1;

        SPTR<Tensor<T / T2>> out = std::make_shared<Tensor<T / T2>>();
        out = MiniTorch<T / T2>::ReShape(out, new_shape);

        // 3 4 5 -1d == 4 3 5 - 0d
        for (int d = 0; d < (T/T2); ++d) {
            SPTR<Element> temp_element = std::make_shared<Element>(.0);
            for (int i = 0; i < T2; ++i) {
                SPTR<Element> take_element = TakeObject(temp_t1, i * temp_t1->getIndexWeights()[0] + d);
                temp_element = AddElement(temp_element, take_element);
            }

            out->SetObject(d, temp_element);
        }

        out = MiniTorch<T / T2>::TakeTranspose(out, dim, 0);

        return out;
    }

    static PTR_E Sum(const SPTR<Tensor<T>> &t1, const int T2, const int offset) {
        const ARR<PTR_E, T> &ref = t1->getData();

        auto out = std::make_shared<Element>(.0);
        for (int i = offset; i < T2 + offset; ++i) {
            out = AddElement(out, ref[i]);
        }

        return out;
    }

    static PTR_E Sum(const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        auto out = std::make_shared<Element>(.0);

        for (const PTR_E& i : ref) {
            out = AddElement(out, i);
        }

        return out;
    }

    static PTR_E Mean(const SPTR<Tensor<T>> &t1, const int T2, const int offset) {
        const ARR<PTR_E, T> &ref = t1->getData();

        auto out = Sum(t1, T2, offset);

        out = MiniTorch<T>::MulElement(out, 1.0l / T2);

        return out;
    }

    static PTR_E Mean(const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        auto out = Sum(t1);

        out = MiniTorch<T>::MulElement(out, 1.0l/T);

        return out;
    }

    static PTR_E Var(const SPTR<Tensor<T>> &t1, const int T2, const int offset) {
        const ARR<PTR_E, T> &ref = t1->getData();

        const auto mean = Mean(t1, T2, offset);
        const auto neg_mean = MulElement(mean, -1);

        auto out = std::make_shared<Element>(.0);
        for (int i = offset; i < T2 + offset; ++i) {
            const PTR_E &temp = ref[i];
            PTR_E copied = temp->Copy();

            copied = MiniTorch<T>::AddElement(copied, neg_mean);
            copied = MiniTorch<T>::Pow(copied, 2);

            out = MiniTorch<T>::AddElement(out, copied);
        }
        out = MiniTorch<T>::MulElement(out, 1.0l/(T2 - 1));

        return out;
    }

    static PTR_E Var(const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        const auto mean = Mean(t1);
        const auto neg_mean = MulElement(mean, -1);

        auto out = std::make_shared<Element>(.0);
        for (int i = 0; i < ref.size(); i++) {
            const PTR_E &temp = ref[i];
            PTR_E copied = temp->Copy();

            copied = MiniTorch<T>::AddElement(copied, neg_mean);
            copied = MiniTorch<T>::Pow(copied, 2);

            out = MiniTorch<T>::AddElement(out, copied);
        }
        out = MiniTorch<T>::MulElement(out, 1.0l/(T - 1));

        return out;
    }

    static PTR_E Std(const SPTR<Tensor<T>> &t1, const int T2, const int offset) {
        return MiniTorch<T>::Pow(Var(t1, T2, offset), 0.5);
    }

    static PTR_E Std(const SPTR<Tensor<T>> &t1) {
        return MiniTorch<T>::Pow(Var(t1), 0.5);
    }

    static PTR_E RMS(const SPTR<Tensor<T>> &t1, const int T2, const int offset) {
        const ARR<PTR_E, T> &ref = t1->getData();

        auto out = std::make_shared<Element>(.0);
        for (int i = offset; i < T2 + offset; ++i) {
            auto copied = MiniTorch<T>::Pow(ref[i]->Copy(), 2);

            out = MiniTorch<T>::AddElement(out, copied);
        }
        out = MiniTorch<T>::MulElement(out, 1.0l / T2);
        out = MiniTorch<T>::Pow(out, 0.5);

        return out;
    }

    static PTR_E RMS(const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        auto out = std::make_shared<Element>(.0);
        for (int i = 0; i < T; ++i) {
            auto copied = MiniTorch<T>::Pow(ref[i]->Copy(), 2);

            out = MiniTorch<T>::AddElement(out, copied);
        }
        out = MiniTorch<T>::MulElement(out, 1.0l / T);
        out = MiniTorch<T>::Pow(out, 0.5);

        return out;
    }

    static VEC_I Argmax(const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        int index = 0;
        for (int i = 1; i < ref.size(); i++) {
            if (ref[i]->getData() > ref[index]->getData()) {
                index = i;
            }
        }

        const VEC_I &out = CreateIndex<T>(t1, index);

        return out;
    }

    static VEC_I Argmin(const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        int index = 0;
        for (int i = 1; i < ref.size(); i++) {
            if (ref[i]->getData() < ref[index]->getData()) {
                index = i;
            }
        }

        const VEC_I &out = CreateIndex<T>(t1, index);

        return out;
    }

    static void PrintTensor(const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        // (B, 1, C, D)

        const VEC_I &shape = t1->getShape();

        cout << "Tensor(";
        // 2, 3, 4 [[
        // 5, 6, 7

        for (int i = 0; i < shape.size()-1; i++) { cout << "["; }

        const int mymax = ref.size();
        for (int i = 0; i < mymax; i++) {
            if (i % (shape[shape.size()-1]) == 0) {
                if (i == 0) {
                    cout << "[";
                }
                else{
                    cout << "        " << "[";
                }
            }

            if ((i+1) % (shape[shape.size()-1]) == 0) { cout << TakeObject<T>(t1, i)->getData(); }
            else { cout << TakeObject<T>(t1, i)->getData() << " "; }

            if ((i+1) % (shape[shape.size()-1]) == 0) {
                cout << "]";
            }

            for (int s = shape.size()-1; s > 0; --s) {
                int cum = 1;
                for (int c = 0; c < s; c++) { cum *= shape[shape.size() - c - 1]; }

                if ((i+1) % cum == 0 && (i+1) != mymax) {
                    cout << endl;
                }
            }
        }

        for (int i = 0; i < shape.size()-1; i++) { cout << "]"; }
        cout << ")" << endl;
    }

    // %% ELEMENT OP. FUNCTIONS %%
    template<LLI T2, LLI T3>
    static SPTR<Tensor<T3>> MatMul(const SPTR<Tensor<T>> &t1, const SPTR<Tensor<T2>> &t2) {
        // [c, a, m] x [c, m, b] -> [c, a,b]
        // [2, 3, 4] x [1, 4, 3] -> Calismali
        // [2, 3, 4] x [5, 4, 3] -> Hata

        const VEC_I &t1_shape = t1->getShape();
        const VEC_I &t2_shape = t2->getShape();

        if (t1_shape[t1_shape.size() - 1] != t2_shape[t2_shape.size() - 2] ||
            t1_shape.size() != t2_shape.size() || t1_shape.size() != 3) {
            throw std::runtime_error("Bad Tensor Shapes!");
        }

        VEC_I a_shape = {-1, t1_shape[t1_shape.size() - 2], t1_shape[t1_shape.size() - 1]};
        VEC_I b_shape = {-1, t2_shape[t2_shape.size() - 2], t2_shape[t2_shape.size() - 1]};

        const SPTR<Tensor<T>> &temp_t1 = MiniTorch<T>::ReShape(t1, a_shape);
        const SPTR<Tensor<T2>> &temp_t2 = MiniTorch<T2>::ReShape(t2, b_shape);

        a_shape = temp_t1->getShape();
        b_shape = temp_t2->getShape();

        bool dynamic = false;
        if (a_shape[0] > b_shape[0] && b_shape[0] == 1) {
            dynamic = true;
        }
        if (a_shape[0] != b_shape[0] && !dynamic) {
            throw std::runtime_error("Bad Tensor Shapes!");
        }

        VEC_I matted_shape = {a_shape[0], a_shape[1], b_shape[2]};

        ARR<SPTR<Element>, T3> out;
        for (int tdim = 0; tdim < matted_shape[0]; ++tdim){
            const int mul_count = a_shape[a_shape.size() - 1];

            for (int r = 0; r < matted_shape[1]; ++r) {
                for (int c = 0; c < matted_shape[2]; ++c) {
                    PTR_E s = std::make_shared<Element>(.0);

                    for (int i = 0; i < mul_count; i++) {
                        s = AddElement(s, MulElement(TakeObject<T>(temp_t1, {tdim, r, i}), TakeObject<T2>(temp_t2, {dynamic ? 0 : tdim, i, c})));
                    }

                    out[tdim * matted_shape[1] * matted_shape[2] + r * matted_shape[2] + c] = s;
                }
            }
        }

        auto t_out = std::make_shared<Tensor<T3>>(out, matted_shape);

        return t_out;
    }

    template<LLI B, LLI T2, LLI T3=4>
    static SPTR<Tensor<T2 * B>> FlexibleMul(const SPTR<Tensor<T>> &t1, const SPTR<Tensor<T2>> &t2, const ARR<int, T3> start, const ARR<int, T3> end) requires (T >= T2) {
        // T1: (B, N, M) | T2: (B, 1, M) - (1, N, M) - (B, N, 1) - (B, 1, 1) - (1, N, 1) - (1, 1, 1) ALL VALID
        ARR<PTR_E, T2 * B> out;

        int t2_counter = 0;
        int t3_counter = 0;
        for (int i = 0; i < T; ++i) {
            if (i % (T / B) == 0) {
                t2_counter = 0;
            }

            VEC_I index = CreateIndex<T>(t1, i); // -> (a, b, c)

            bool skip = false;
            for (int i2 = 0; i2 < index.size(); ++i2) {
                if ((index[i2] > end[i2]) || (index[i2] < start[i2])) {
                    skip = true;
                    break;
                }
            }

            if (skip) {
                continue;
            }

            const PTR_E &a_element = TakeObject<T>(t1, i);
            const PTR_E &b_element = TakeObject<T2 >(t2, t2_counter);

            out[t3_counter] = MulElement(a_element, b_element);

            ++t2_counter;
            ++t3_counter;
        }

        VEC_I new_shape = t2->getShape();
        new_shape[0] = B;

        return std::make_shared<Tensor<T2 * B>>(out, new_shape);
    }
    template<LLI T2>
    static SPTR<Tensor<T>> FlexibleMul(const SPTR<Tensor<T>> &t1, const SPTR<Tensor<T2>> &t2) requires (T >= T2) {
        // T1: (B, N, M) | T2: (B, 1, M) - (1, N, M) - (B, N, 1) - (B, 1, 1) - (1, N, 1) - (1, 1, 1) ALL VALID

        const ARR<PTR_E, T> &ref = t1->getData();
        // const ARR<PTR_E, T> &ref2 = t2->getData();

        ARR<PTR_E, T> out;

        const VEC_I &a_shape = t1->getShape();
        const VEC_I &b_shape = t2->getShape();

        for (int i = 0; i < ref.size(); i++) {
            VEC_I index = CreateIndex<T>(t1, i); // -> (a, b, c)

            VEC_I free_index;
            free_index.reserve(index.size());
            for (int t = 0; t < index.size(); t++) {
                if (a_shape[t] == b_shape[t]) { free_index.push_back(index[t]); }
                else { free_index.push_back(0); }
            } // (a, b, c) -> if its free take "a" but if dim=1 is not free take "0" -> (1, b, c)

            const PTR_E &a_element = TakeObject<T>(t1, index);
            const PTR_E &b_element = TakeObject<T2>(t2, free_index);

            out[i] = MulElement(a_element, b_element);
        }

        return std::make_shared<Tensor<T>>(out, a_shape);
    }
    template<LLI T2>
    static SPTR<Tensor<T2>> FlexibleMul(const SPTR<Tensor<T>> &t1, const SPTR<Tensor<T2>> &t2) requires (T2 > T) {
        // T1: (B, N, M) | T2: (B, 1, M) - (1, N, M) - (B, N, 1) - (B, 1, 1) - (1, N, 1) - (1, 1, 1) ALL VALID

        const ARR<PTR_E, T2> &ref = t2->getData();
        // const ARR<PTR_E, T> &ref2 = t2->getData();

        ARR<PTR_E, T2> out;

        const VEC_I &a_shape = t1->getShape();
        const VEC_I &b_shape = t2->getShape();

        for (int i = 0; i < ref.size(); i++) {
            VEC_I index = CreateIndex<T2>(t2, i); // -> (a, b, c)

            VEC_I free_index;
            free_index.reserve(index.size());
            for (int t = 0; t < index.size(); t++) {
                if (a_shape[t] == b_shape[t]) { free_index.push_back(index[t]); }
                else { free_index.push_back(0); }
            } // (a, b, c) -> if its free take "a" but if dim=1 is not free take "0" -> (1, b, c)

            const PTR_E &a_element = TakeObject<T>(t1, free_index);
            const PTR_E &b_element = TakeObject<T2>(t2, index);

            out[i] = MulElement(a_element, b_element);
        }

        return std::make_shared<Tensor<T2>>(out, a_shape);
    }
    template<LLI T2>
    static SPTR<Tensor<T>> FlexibleMul(const SPTR<Tensor<T2>> &t2, const SPTR<Tensor<T>> &t1) requires (T > T2) {
        // T1: (B, N, M) | T2: (B, 1, M) - (1, N, M) - (B, N, 1) - (B, 1, 1) - (1, N, 1) - (1, 1, 1) ALL VALID

        const ARR<PTR_E, T> &ref = t1->getData();
        // const ARR<PTR_E, T> &ref2 = t2->getData();

        ARR<PTR_E, T> out;

        const VEC_I &a_shape = t1->getShape();
        const VEC_I &b_shape = t2->getShape();

        for (int i = 0; i < ref.size(); i++) {
            VEC_I index = CreateIndex<T>(t1, i); // -> (a, b, c)

            VEC_I free_index;
            free_index.reserve(index.size());
            for (int t = 0; t < index.size(); t++) {
                if (a_shape[t] == b_shape[t]) { free_index.push_back(index[t]); }
                else { free_index.push_back(0); }
            } // (a, b, c) -> if its free take "a" but if dim=1 is not free take "0" -> (1, b, c)

            const PTR_E &a_element = TakeObject<T>(t1, index);
            const PTR_E &b_element = TakeObject<T2>(t2, free_index);

            out[i] = MulElement(a_element, b_element);
        }

        return std::make_shared<Tensor<T>>(out, a_shape);
    }
    template<LLI T2>
    static SPTR<Tensor<T2>> FlexibleMul(const SPTR<Tensor<T2>> &t2, const SPTR<Tensor<T>> &t1) requires (T2 > T) {
        // T1: (B, N, M) | T2: (B, 1, M) - (1, N, M) - (B, N, 1) - (B, 1, 1) - (1, N, 1) - (1, 1, 1) ALL VALID

        const ARR<PTR_E, T2> &ref = t2->getData();
        // const ARR<PTR_E, T> &ref2 = t2->getData();

        ARR<PTR_E, T2> out;

        const VEC_I &a_shape = t1->getShape();
        const VEC_I &b_shape = t2->getShape();

        for (int i = 0; i < ref.size(); i++) {
            VEC_I index = CreateIndex<T2>(t2, i); // -> (a, b, c)

            VEC_I free_index;
            free_index.reserve(index.size());
            for (int t = 0; t < index.size(); t++) {
                if (a_shape[t] == b_shape[t]) { free_index.push_back(index[t]); }
                else { free_index.push_back(0); }
            } // (a, b, c) -> if its free take "a" but if dim=1 is not free take "0" -> (1, b, c)

            const PTR_E &a_element = TakeObject<T>(t1, free_index);
            const PTR_E &b_element = TakeObject<T2>(t2, index);

            out[i] = MulElement(a_element, b_element);
        }

        return std::make_shared<Tensor<T2>>(out, a_shape);
    }
    static SPTR<Tensor<T>> FlexibleMul(const SPTR<Tensor<T>> &t1, const DTYPE b) {
        const ARR<PTR_E, T> &ref = t1->getData();

        ARR<PTR_E, T> out;
        for (int i = 0; i < ref.size(); i++) {
            out[i] = MulElement(ref[i], b);
        }

        return std::make_shared<Tensor<T>>(out, t1->getShape());
    }
    static SPTR<Tensor<T>> FlexibleMul(const DTYPE b, const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        ARR<PTR_E, T> out;
        for (int i = 0; i < ref.size(); i++) {
            out[i] = MulElement(ref[i], b);
        }

        return std::make_shared<Tensor<T>>(out, t1->getShape());
    }

    template<LLI T2>
    static SPTR<Tensor<T>> FlexibleDiv(const SPTR<Tensor<T>> &t1, const SPTR<Tensor<T2>> &t2) requires (T > T2) {
        const ARR<PTR_E, T> &ref = t1->getData();

        ARR<PTR_E, T> out;

        const VEC_I &a_shape = t1->getShape();
        const VEC_I &b_shape = t2->getShape();

        for (int i = 0; i < ref.size(); i++) {
            VEC_I index = CreateIndex<T>(t1, i); // -> (a, b, c)

            VEC_I free_index;
            free_index.reserve(index.size());
            for (int t = 0; t < index.size(); t++) {
                if (a_shape[t] == b_shape[t]) { free_index.push_back(index[t]); }
                else { free_index.push_back(0); }
            } // (a, b, c) -> if its free take "a" but if dim=1 is not free take "0" -> (1, b, c)

            const PTR_E &a_element = TakeObject<T>(t1, index);
            const PTR_E &b_element = TakeObject<T2>(t2, free_index);

            out[i] = DivElement(a_element, b_element);
        }

        return std::make_shared<Tensor<T>>(out, a_shape);
    }
    static SPTR<Tensor<T>> FlexibleDiv(const SPTR<Tensor<T>> &t1, const DTYPE b) {
        const ARR<PTR_E, T> &ref = t1->getData();

        ARR<PTR_E, T> out;
        for (int i = 0; i < ref.size(); i++) {
            out[i] = DivElement(ref[i], b);
        }

        return std::make_shared<Tensor<T>>(out, t1->getShape());
    }
    static SPTR<Tensor<T>> FlexibleDiv(const DTYPE b, const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        ARR<PTR_E, T> out;
        for (int i = 0; i < ref.size(); i++) {
            out[i] = DivElement(b, ref[i]);
        }

        return std::make_shared<Tensor<T>>(out, t1->getShape());
    }

    template<LLI T2>
    static SPTR<Tensor<T>> FlexibleAdd(const SPTR<Tensor<T>> &t1, const SPTR<Tensor<T2>> &t2) requires (T >= T2) {
        const ARR<PTR_E, T> &ref = t1->getData();

        ARR<PTR_E, T> out;

        const VEC_I &a_shape = t1->getShape();
        const VEC_I &b_shape = t2->getShape();

        std::vector<bool> free_dims;
        for (int i = 0; i < a_shape.size(); i++) { free_dims.push_back(a_shape[i] == b_shape[i]); }

        for (int i = 0; i < ref.size(); i++) {
            VEC_I index = CreateIndex<T>(t1, i); // -> (a, b, c)

            VEC_I free_index;
            free_index.reserve(index.size());
            for (int t = 0; t < index.size(); t++) {
                if (free_dims[t]) {
                    free_index.push_back(index[t]);
                }
                else {
                    free_index.push_back(0);
                }
            } // (a, b, c) -> if its free take "a" but if dim=1 is not free take "0" -> (1, b, c)

            const PTR_E &a_element = TakeObject<T>(t1, index);
            const PTR_E &b_element = TakeObject<T2>(t2, free_index);

            out[i] = AddElement(a_element, b_element);
        }

        return std::make_shared<Tensor<T>>(out, a_shape, t1->getIndexWeights(), t1->getComplex());
    }
    template<LLI T2>
    static SPTR<Tensor<T2>> FlexibleAdd(const SPTR<Tensor<T>> &t1, const SPTR<Tensor<T2>> &t2) requires (T2 > T) {
        const ARR<PTR_E, T2> &ref = t2->getData();

        ARR<PTR_E, T2> out;

        const VEC_I &a_shape = t1->getShape();
        const VEC_I &b_shape = t2->getShape();

        std::vector<bool> free_dims;
        for (int i = 0; i < a_shape.size(); i++) { free_dims.push_back(a_shape[i] == b_shape[i]); }

        for (int i = 0; i < ref.size(); i++) {
            VEC_I index = CreateIndex<T2>(t2, i); // -> (a, b, c)

            VEC_I free_index;
            free_index.reserve(index.size());
            for (int t = 0; t < index.size(); t++) {
                if (free_dims[t]) {
                    free_index.push_back(index[t]);
                }
                else {
                    free_index.push_back(0);
                }
            } // (a, b, c) -> if its free take "a" but if dim=1 is not free take "0" -> (1, b, c)

            const PTR_E &a_element = TakeObject<T>(t1, free_index);
            const PTR_E &b_element = TakeObject<T2>(t2, index);

            out[i] = AddElement(a_element, b_element);
        }

        return std::make_shared<Tensor<T2>>(out, a_shape, t1->getIndexWeights(), t1->getComplex());
    }
    template<LLI T2>
    static SPTR<Tensor<T>> FlexibleAdd(const SPTR<Tensor<T2>> &t2, const SPTR<Tensor<T>> &t1) requires (T > T2) {
        const ARR<PTR_E, T> &ref = t1->getData();

        ARR<PTR_E, T> out;

        const VEC_I &a_shape = t1->getShape();
        const VEC_I &b_shape = t2->getShape();

        std::vector<bool> free_dims;
        for (int i = 0; i < a_shape.size(); i++) { free_dims.push_back(a_shape[i] == b_shape[i]); }

        for (int i = 0; i < ref.size(); i++) {
            VEC_I index = CreateIndex<T>(t1, i); // -> (a, b, c)

            VEC_I free_index;
            free_index.reserve(index.size());
            for (int t = 0; t < index.size(); t++) {
                if (free_dims[t]) {
                    free_index.push_back(index[t]);
                }
                else {
                    free_index.push_back(0);
                }
            } // (a, b, c) -> if its free take "a" but if dim=1 is not free take "0" -> (1, b, c)

            const PTR_E &a_element = TakeObject<T>(t1, index);
            const PTR_E &b_element = TakeObject<T2>(t2, free_index);

            out[i] = AddElement(a_element, b_element);
        }

        return std::make_shared<Tensor<T>>(out, a_shape, t1->getIndexWeights(), t1->getComplex());
    }
    template<LLI T2>
    static SPTR<Tensor<T2>> FlexibleAdd(const SPTR<Tensor<T2>> &t2, const SPTR<Tensor<T>> &t1) requires (T2 > T) {
        const ARR<PTR_E, T2> &ref = t2->getData();

        ARR<PTR_E, T2> out;

        const VEC_I &a_shape = t1->getShape();
        const VEC_I &b_shape = t2->getShape();

        std::vector<bool> free_dims;
        for (int i = 0; i < a_shape.size(); i++) { free_dims.push_back(a_shape[i] == b_shape[i]); }

        for (int i = 0; i < ref.size(); i++) {
            VEC_I index = CreateIndex<T2>(t2, i); // -> (a, b, c)

            VEC_I free_index;
            free_index.reserve(index.size());
            for (int t = 0; t < index.size(); t++) {
                if (free_dims[t]) {
                    free_index.push_back(index[t]);
                }
                else {
                    free_index.push_back(0);
                }
            } // (a, b, c) -> if its free take "a" but if dim=1 is not free take "0" -> (1, b, c)

            const PTR_E &a_element = TakeObject<T>(t1, free_index);
            const PTR_E &b_element = TakeObject<T2>(t2, index);

            out[i] = AddElement(a_element, b_element);
        }

        return std::make_shared<Tensor<T2>>(out, a_shape, t1->getIndexWeights(), t1->getComplex());
    }
    static SPTR<Tensor<T>> FlexibleAdd(const SPTR<Tensor<T>> &t1, const DTYPE b) {
        const ARR<PTR_E, T> &ref = t1->getData();

        ARR<PTR_E, T> out;
        for (int i = 0; i < ref.size(); i++) {
            out[i] = AddElement(ref[i], b);
        }

        return std::make_shared<Tensor<T>>(out, t1->getShape(), t1->getIndexWeights(), t1->getComplex());
    }
    static SPTR<Tensor<T>> FlexibleAdd(const DTYPE b, const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        ARR<PTR_E, T> out;
        for (int i = 0; i < ref.size(); i++) {
            out[i] = AddElement(ref[i], b);
        }

        return std::make_shared<Tensor<T>>(out, t1->getShape(), t1->getIndexWeights(), t1->getComplex());
    }

    template<LLI T2>
    static SPTR<Tensor<T>> FlexibleSub(const SPTR<Tensor<T>> &t1, const SPTR<Tensor<T2>> &t2) requires (T >= T2) {
        const ARR<PTR_E, T> &ref = t1->getData();

        ARR<PTR_E, T> out;

        const VEC_I &a_shape = t1->getShape();
        const VEC_I &b_shape = t2->getShape();

        std::vector<bool> free_dims;
        for (int i = 0; i < a_shape.size(); i++) { free_dims.push_back(a_shape[i] == b_shape[i]); }

        for (int i = 0; i < ref.size(); i++) {
            VEC_I index = CreateIndex<T>(t1, i); // -> (a, b, c)

            VEC_I free_index;
            free_index.reserve(index.size());
            for (int t = 0; t < index.size(); t++) {
                if (free_dims[t]) {
                    free_index.push_back(index[t]);
                }
                else {
                    free_index.push_back(0);
                }
            } // (a, b, c) -> if its free take "a" but if dim=1 is not free take "0" -> (1, b, c)

            const PTR_E &a_element = TakeObject<T>(t1, index);
            const PTR_E &b_element = TakeObject<T2>(t2, free_index);

            out[i] = SubElement(a_element, b_element);
        }

        return std::make_shared<Tensor<T>>(out, a_shape, t1->getIndexWeights(), t1->getComplex());
    }
    static SPTR<Tensor<T>> FlexibleSub(const SPTR<Tensor<T>> &t1, const DTYPE b) {
        const ARR<PTR_E, T> &ref = t1->getData();

        ARR<PTR_E, T> out;
        for (int i = 0; i < ref.size(); i++) {
            out[i] = SubElement(ref[i], b);
        }

        return std::make_shared<Tensor<T>>(out, t1->getShape());
    }
    static SPTR<Tensor<T>> FlexibleSub(const DTYPE b, const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        ARR<PTR_E, T> out;
        for (int i = 0; i < ref.size(); i++) {
            out[i] = SubElement(b, ref[i]);
        }

        return std::make_shared<Tensor<T>>(out, t1->getShape());
    }
    
    static PTR_E MulElement(const PTR_E &e1, const PTR_E &e2) {
        return std::make_shared<Element>(e1->getData() * e2->getData(), e1, e2->getData(), e2, e1->getData());
    }
    static PTR_E MulElement(const PTR_E &e1, DTYPE scalar) {
        return std::make_shared<Element>(e1->getData() * scalar, e1, scalar);
    }

    static PTR_E DivElement(const PTR_E &e1, const PTR_E &e2) {
        return std::make_shared<Element>(e1->getData() / e2->getData(), e1, 1/e2->getData(), e2, (-e1->getData()/std::pow(e2->getData(), 2)));
    }
    static PTR_E DivElement(const PTR_E &e1, const DTYPE scalar) {
        return std::make_shared<Element>(e1->getData() / scalar, e1, 1/scalar);
    }
    static PTR_E DivElement(const DTYPE scalar, const PTR_E &e1) {
        return std::make_shared<Element>(scalar / e1->getData(), e1, -1/pow(e1->getData(), 2));
    }

    static PTR_E AddElement(const PTR_E &e1, const PTR_E &e2) {
        return std::make_shared<Element>(e1->getData() + e2->getData(), e1, 1.0, e2, 1.0);
    }
    static PTR_E AddElement(const PTR_E &e1, const DTYPE scalar) {
        return std::make_shared<Element>(e1->getData() + scalar, e1, 1.0);
    }

    static PTR_E SubElement(const PTR_E &e1, const PTR_E &e2) {
        return std::make_shared<Element>(e1->getData() - e2->getData(), e1, 1.0, e2, -1.0);
    }
    static PTR_E SubElement(const PTR_E &e1, const DTYPE scalar) {
        return std::make_shared<Element>(e1->getData() - scalar, e1, 1.0);
    }
    static PTR_E SubElement(const DTYPE scalar, const PTR_E &e1) {
        return std::make_shared<Element>(scalar - e1->getData(), e1, -1.0);
    }

    static SPTR<Tensor<T>> Pow(const SPTR<Tensor<T>> &t1, const DTYPE scalar) {
        const ARR<PTR_E, T> &ref = t1->getData();

        const SPTR<Tensor<T>> &temp = t1->Copy();
        for (int i = 0; i < ref.size(); i++) {
            temp->getData()[i] = Pow(temp->getData()[i], scalar);
        }

        return temp;
    }
    static PTR_E Pow(const PTR_E &e1, const DTYPE scalar) {
        return std::make_shared<Element>(std::pow(e1->getData(), scalar), e1, scalar * pow(e1->getData(), scalar-1));
    }

    static SPTR<Tensor<T>> Square(const SPTR<Tensor<T>> &t1){ return Pow(t1, 2); }
    static PTR_E Square(const PTR_E &e1){ return Pow(e1, 2); }

    static SPTR<Tensor<T>> Sqrt(const SPTR<Tensor<T>> &t1) { return Pow(t1, 0.5); }
    static PTR_E Sqrt(const PTR_E &e1) { return Pow(e1, 0.5); }

    static SPTR<Tensor<T>> Abs(const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        const SPTR<Tensor<T>> &temp = t1->Copy();
        for (int i = 0; i < ref.size(); i++) {
            temp->getData()[i] = Abs(temp->getData()[i]);
        }

        return temp;
    }
    static PTR_E Abs(const PTR_E &e1) {
        if (e1->getData() < 0) {
            return std::make_shared<Element>(e1->getData() * -1, e1->getImData(), .0, e1->Copy(), -1);
        }
        return std::make_shared<Element>(e1->getData(), e1->getImData(), .0, e1->Copy(), 1);
    }

    static SPTR<Tensor<T>> Exp(const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        const SPTR<Tensor<T>> &temp = t1->Copy();
        for (int i = 0; i < ref.size(); i++) {
            temp->getData()[i] = Exp(temp->getData()[i]);
        }

        return temp;
    }
    static PTR_E Exp(const PTR_E &e1) {
        return std::make_shared<Element>(std::pow(EULER, e1->getData()), e1, std::pow(EULER, e1->getData()));
    }

    static SPTR<Tensor<T>> Log(const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        const SPTR<Tensor<T>> &temp = t1->Copy();
        for (int i = 0; i < ref.size(); i++) {
            temp->getData()[i] = Log(temp->getData()[i]);
        }

        return temp;
    }
    static PTR_E Log(const PTR_E &e1) {
        return std::make_shared<Element>(std::log(e1->getData()), e1, 1/e1->getData());
    }

    // %% ACTIVATIONS %%
    static SPTR<Tensor<T>> Softplus(const SPTR<Tensor<T>> &t1, const DTYPE beta=1, const DTYPE treshold=20) {
        const ARR<PTR_E, T> &ref = t1->getData();

        const SPTR<Tensor<T>> &temp = t1->Copy();
        for (int i = 0; i < ref.size(); i++) {
            temp->getData()[i] = Softplus(temp->getData()[i], beta, treshold);
        }

        return temp;
    }
    static PTR_E Softplus(const PTR_E &e1, const DTYPE beta, const DTYPE treshold=20) {
        if (e1->getData() >= treshold) {
            return e1;
        }
        &e1 = MulElement(Log(AddElement(Exp(MulElement(e1, beta)),1)),1/beta);

        return e1;
    }

    template<int D>
    static SPTR<Tensor<T>> Softmax(const SPTR<Tensor<T>> &t1) {
        // T : (A, B, C,...) -> (T / DIM, DIM) Bu fonksiyon tek batch tek channel icin
        // Copy : (T/D, D) -> exp(Ni)/sum(exp(N)) -> (B, N)
        SPTR<Tensor<T>> tensor_copy = t1->Copy();

        const auto &old_shape = t1->getShape();

        VEC_I soShape = {-1, D};
        tensor_copy = ReShape(tensor_copy, soShape);

        int f = 0;
        ARR<PTR_E, T> soft_vec;
        for (int b = 0; b < T / D; b++) {
            std::vector<PTR_E> exps = {};
            for (int n = b * D; n < b * D + D; ++n) {
                auto temp_element = tensor_copy->TakeObject(n);
                exps.push_back(Exp(temp_element));
            }

            PTR_E sum = std::make_shared<Element>(1e-9);
            for (PTR_E e : exps) {
                sum = AddElement(sum, e);
            }

            for (PTR_E e : exps) {
                e = DivElement(e, sum);
                soft_vec[f] = e;
                ++f;
            }
        }

        return std::make_shared<Tensor<T>>(soft_vec, old_shape);
    }

    static SPTR<Tensor<T>> Sigmoid(const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        const SPTR<Tensor<T>> &temp = t1->Copy();
        for (int i = 0; i < ref.size(); i++) {
            auto ti = Sigmoid(temp->getData()[i]);

            temp->SetObject(i, ti);
        }

        return temp;
    }
    static PTR_E Sigmoid(const PTR_E &e1) {
        const PTR_E &tempexp = Exp(e1);
        const PTR_E &add = AddElement(tempexp, 1.0);

        return DivElement(tempexp, add);
    }

    static SPTR<Tensor<T>> ReLU(const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        const SPTR<Tensor<T>> &temp = t1->Copy();
        for (int i = 0; i < ref.size(); i++) {
            temp->getData()[i] = ReLU(temp->getData()[i]);
        }

        return temp;
    }
    static PTR_E ReLU(const PTR_E &e1) {
        if (e1->getData() > 0) {
            return Abs(e1);
        }
        return std::make_shared<Element>(.0,.0);
    }

    static SPTR<Tensor<T>> GeLU(const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        const SPTR<Tensor<T>> &temp = t1->Copy();
        for (int i = 0; i < ref.size(); i++) {
            temp->getData()[i] = GeLU(temp->getData()[i]);
        }

        return temp;
    }
    static PTR_E GeLU(const PTR_E &e1) {
        const PTR_E &temp = MulElement(MulElement(AddElement(
                                                    Tanh(
                                                        MulElement(
                                                            AddElement(
                                                                MulElement(
                                                                    Pow(e1, 3),
                                                                    0.044715),
                                                                e1),
                                                            std::pow((2.0/PI), .5))),
                                                    1.0), e1), .5);

        return temp;
    }

    static SPTR<Tensor<T>> SiLU(const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        const SPTR<Tensor<T>> &temp = t1->Copy();
        for (int i = 0; i < ref.size(); i++) {
            temp->getData()[i] = SiLU(temp->getData()[i]);
        }

        return temp;
    }
    static PTR_E SiLU(const PTR_E &e1) {
        const PTR_E &silued = Sigmoid(e1);

        return MulElement(silued, e1);
    }

    static SPTR<Tensor<T>> Tanh(const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        const SPTR<Tensor<T>> &temp = t1->Copy();
        for (int i = 0; i < ref.size(); i++) {
            temp->getData()[i] = Tanh(temp->getData()[i]);
        }

        return temp;
    }
    static PTR_E Tanh(const PTR_E &e1) {
        const PTR_E &nom = MulElement(SubElement(Exp(MulElement(e1, -2)), 1), -1);
        const PTR_E &denom = AddElement(Exp(MulElement(e1, -2)), 1);

        const PTR_E &result = DivElement(nom, denom);

        return result;
    }

    // %% LOSSES %%
    static SPTR<Tensor<T>> CategoricalCrossEntropy(const SPTR<Tensor<T>> &t1, const int index) {
        // probs : (B, N) -> N = Number of Classes
        const int class_index = index < 0 ? index + t1->getShape()[1] : index;

        ARR<PTR_E, T> loss_vec;
        for (int i = 0; i < t1->getShape()[0]; i++) {
            VEC_I start{i, 0};
            VEC_I end{i, t1->getShape()[0]-1};

            PTR_E &sliced = t1->TakeSlice(start, end)->getData()[class_index]; // (N,)[Class_index] -> score
            &sliced = MulElement(Log(AddElement(sliced, 0.00001)), -1);

            loss_vec[i] = sliced;
        }

        return std::make_shared<Tensor<T>>(loss_vec);
    }
    static PTR_E CategoricalCrossEntropy(const std::vector<PTR_E> &probs, const int index) {
        const int class_index = index < 0 ? index + probs.size() : index;

        return MulElement(Log(AddElement(probs[class_index], 0.00001)), -1);
    }

    static SPTR<Tensor<T>> BinaryCrossEntropy(const SPTR<Tensor<T>> &t1, const int index) {
        // t1 : (B, 2)
        const int class_index = index < 0 ? index + t1->getShape()[1] : index;
        return CategoricalCrossEntropy(t1, class_index);
    }
    static PTR_E BinaryCrossEntropy(const std::vector<PTR_E> &probs, const int index) {
        const int class_index = index < 0 ? index + probs.size() : index;
        return CategoricalCrossEntropy(probs, class_index);
    }

    template<LLI T2>
    static SPTR<Tensor<T>> MSE(const SPTR<Tensor<T>> &t1, const SPTR<Tensor<T2>> &t2) requires (T >= T2) {
        return Pow(FlexibleSub(t1, t2), 2);
    }
    template<LLI T2>
    static SPTR<Tensor<T2>> MSE(const SPTR<Tensor<T>> &t1, const SPTR<Tensor<T2>> &t2) requires (T2 > T){
        return Pow(FlexibleSub(t1, t2), 2);
    }
    static PTR_E MSE(const PTR_E &e1, const DTYPE y) {
        return Pow(SubElement(e1, y), 2);;
    }

    template<LLI T2>
    static SPTR<Tensor<T>> MAE(const SPTR<Tensor<T>> &t1, const SPTR<Tensor<T2>> &t2) requires (T >= T2){
        return Abs(FlexibleSub(t1, t2));
    }
    template<LLI T2>
    static SPTR<Tensor<T2>> MAE(const SPTR<Tensor<T>> &t1, const SPTR<Tensor<T2>> &t2) requires (T2 > T){
        return Abs(FlexibleSub(t1, t2));
    }
    static PTR_E MAE(const PTR_E &e1, const DTYPE y) {
        e1 = Abs(SubElement(e1, y));

        return e1;
    }

    // %% UTILITY FUNCTIONS %%
    static void Save(String path, const SPTR<Tensor<T>> &obj) {
        if (path.size() < 1) {
            throw std::runtime_error("Bad Path!");
        }
        if (path.substr(path.size() - 3, path.size()) == EXT && path.size() == 3) {
            throw std::runtime_error("Bad Path!");
        }
        if (path.substr(path.size() - 3, path.size()) != EXT) {
            path += EXT;
        }

        std::ofstream file(path);

        if (!file.is_open()) {
            throw std::runtime_error("File Issue!");
        }

        String shape_str;
        String weights_str;
        String complex_str = std::to_string(obj->getComplex());
        String data_str;

        for (int i = 0; i < obj->getShape().size(); i++) {
            shape_str += std::to_string(obj->getShape()[i]);
            shape_str += '|';

            weights_str += std::to_string(obj->getIndexWeights()[i]);
            weights_str += '|';
        }

        for (int i = 0; i < obj->getData().size(); i++) {
            data_str += std::to_string(obj->getData()[i]->getData());
            data_str += '?';
            data_str += std::to_string(obj->getData()[i]->getImData());
            data_str += '?';
            data_str += std::to_string(obj->getData()[i]->getGradient());
            data_str += '?';
            data_str += '|';
        }

        // cout << "%START_COMPLEX%   " << complex_str << "   %START_SHAPE%   " << shape_str << "   %START_WEIGHTS%   " << weights_str <<  "   %START_DATA%   " << data_str;

        file << complex_str << "\n" << shape_str << "\n" << weights_str << "\n" << std::to_string(obj->getData().size()) << "\n" << data_str << "\n";
    }

    static SPTR<Tensor<T>> Load(String path) {
        std::ifstream file(path);

        if (!file.is_open()) {
            throw std::runtime_error("File Issue!");
        }

        const String s((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        std::vector<String> elements = SplitByChar(s, '\n');

        bool complex = elements[0] == "1";
        VEC_I shape_vec;
        VEC_I weights_vec;
        ARR<PTR_E, T> data_vec;

        std::vector<String> shape_str = SplitByChar(elements[1], '|');
        for (int i = 0; i < shape_str.size(); i++) {
            shape_vec.push_back(std::stoi(shape_str[i]));
        }

        std::vector<String> weights_str = SplitByChar(elements[2], '|');
        for (int i = 0; i < weights_str.size(); i++) {
            weights_vec.push_back(std::stoi(weights_str[i]));
        }

        std::vector<String> data_str = SplitByChar(elements[4], '|');
        for (int i = 0; i < data_str.size(); i++) {
            std::vector<String> element_str = SplitByChar(data_str[i], '?');

            DTYPE data = std::stold(element_str[0]);
            DTYPE im_data = std::stold(element_str[1]);
            DTYPE gradient = std::stold(element_str[2]);

            data_vec[i] = std::make_shared<Element>(data, im_data, gradient);
        }

        return std::make_shared<Tensor<T>>(data_vec, shape_vec, weights_vec, complex);
    }

    template<LLI B, LLI C, LLI H, LLI W, LLI P_h, LLI P_w>
    static SPTR<Tensor<B * C * (P_w + W) * (P_h + H)>> PadInput(SPTR<Tensor<T>> x) requires (P_w > 1 && P_h > 1) {
        constexpr int leftSize = P_w / 2;
        constexpr int upSize = P_h / 2;

        constexpr int downSize = P_h - upSize;
        constexpr int rightSize = P_w - leftSize;

        constexpr int _upSize = upSize * B * C * (W + rightSize);
        constexpr int _leftSize = leftSize * B * C * (H + P_h);

        constexpr int _downSize = downSize * B * C * W;
        constexpr int _rightSize = rightSize * B * C * (H + downSize);

        auto down_pad = MiniTorch<downSize>::Zeros();
        down_pad = MiniTorch<downSize>::ReShape(down_pad, {1, 1, downSize, 1}); // {1, 1, DOWNSIZE, 1}
        auto repped_down_pad = MiniTorch<downSize>::template Repeat<downSize * B * C>(down_pad, {B, C, 1, 1});
        auto full_down_pad = MiniTorch<downSize * B * C>::template Repeat<downSize * B * C * W>(repped_down_pad, {1, 1, 1, W}); // {B, C, DOWNSIZE, W}

        auto right_pad = MiniTorch<rightSize>::Zeros();
        right_pad = MiniTorch<rightSize>::ReShape(right_pad, {1, 1, 1, rightSize}); // {1, 1, 1, RIGHTSIZE}
        auto repped_right_pad = MiniTorch<rightSize>::template Repeat<rightSize * B * C>(right_pad, {B, C, 1, 1});
        auto full_right_pad = MiniTorch<rightSize * B * C>::template Repeat<rightSize * B * C * (H + downSize)>(repped_right_pad, {1, 1, (H + downSize), 1}); // {B, C, DOWNSIZE, W}

        auto up_pad = MiniTorch<upSize>::Zeros();
        up_pad = MiniTorch<upSize>::ReShape(up_pad, {1, 1, upSize, 1});
        auto repped_up_pad = MiniTorch<upSize>::template Repeat<upSize * B * C>(up_pad, {B, C, 1, 1});
        auto full_up_pad = MiniTorch<upSize * B * C>::template Repeat<upSize * B * C * (W + rightSize)>(repped_up_pad, {1, 1, 1, (W + rightSize)}); // {B, C, UPSIZE, W}

        auto left_pad = MiniTorch<leftSize>::Zeros();
        left_pad = MiniTorch<leftSize>::ReShape(left_pad, {1, 1, 1, leftSize});
        auto repped_left_pad = MiniTorch<leftSize>::template Repeat<leftSize * B * C>(left_pad, {B, C, 1, 1});
        auto full_left_pad = MiniTorch<leftSize * B * C>::template Repeat<leftSize * B * C * (H + P_h)>(repped_left_pad, {1, 1, H + P_h, 1}); // {B, C, H, RIGHTSIZE}

        auto downed = MiniTorch<T>::template Concatenate<_downSize>(x, full_down_pad, 2);
        auto righted = MiniTorch<T + _downSize>::template Concatenate<_rightSize>(downed, full_right_pad, 3);
        auto upped = MiniTorch<_upSize>::template Concatenate<T + _downSize + _rightSize>(full_up_pad, righted, 2);
        auto lefted = MiniTorch<_leftSize>::template Concatenate<T + _downSize + _rightSize + _upSize>(full_left_pad, upped, 3);

        return lefted;
    }
    template<LLI B, LLI C, LLI H, LLI W, LLI P_h, LLI P_w>
    static SPTR<Tensor<B * C * (P_w + W) * (P_h + H)>> PadInput(SPTR<Tensor<T>> x) requires (P_w == 1 && P_h > 1) {
        constexpr int leftSize = P_w / 2;
        constexpr int upSize = P_h / 2;

        constexpr int downSize = P_h - upSize;
        constexpr int rightSize = P_w - leftSize;

        constexpr int _upSize = upSize * B * C * (W + rightSize);
        // constexpr int _leftSize = leftSize * B * C * (H + P_h);

        constexpr int _downSize = downSize * B * C * W;
        constexpr int _rightSize = rightSize * B * C * (H + downSize);

        auto down_pad = MiniTorch<downSize>::Zeros();
        down_pad = MiniTorch<downSize>::ReShape(down_pad, {1, 1, downSize, 1}); // {1, 1, DOWNSIZE, 1}
        auto repped_down_pad = MiniTorch<downSize>::template Repeat<downSize * B * C>(down_pad, {B, C, 1, 1});
        auto full_down_pad = MiniTorch<downSize * B * C>::template Repeat<downSize * B * C * W>(repped_down_pad, {1, 1, 1, W}); // {B, C, DOWNSIZE, W}

        auto right_pad = MiniTorch<rightSize>::Zeros();
        right_pad = MiniTorch<rightSize>::ReShape(right_pad, {1, 1, 1, rightSize}); // {1, 1, 1, RIGHTSIZE}
        auto repped_right_pad = MiniTorch<rightSize>::template Repeat<rightSize * B * C>(right_pad, {B, C, 1, 1});
        auto full_right_pad = MiniTorch<rightSize * B * C>::template Repeat<rightSize * B * C * (H + downSize)>(repped_right_pad, {1, 1, (H + downSize), 1}); // {B, C, DOWNSIZE, W}

        auto up_pad = MiniTorch<upSize>::Zeros();
        up_pad = MiniTorch<upSize>::ReShape(up_pad, {1, 1, upSize, 1});
        auto repped_up_pad = MiniTorch<upSize>::template Repeat<upSize * B * C>(up_pad, {B, C, 1, 1});
        auto full_up_pad = MiniTorch<upSize * B * C>::template Repeat<upSize * B * C * (W + rightSize)>(repped_up_pad, {1, 1, 1, (W + rightSize)}); // {B, C, UPSIZE, W}

        auto downed = MiniTorch<T>::template Concatenate<_downSize>(x, full_down_pad, 2);
        auto righted = MiniTorch<T + _downSize>::template Concatenate<_rightSize>(downed, full_right_pad, 3);
        auto upped = MiniTorch<_upSize>::template Concatenate<T + _downSize + _rightSize>(full_up_pad, righted, 2);;

        return upped;
    }
    template<LLI B, LLI C, LLI H, LLI W, LLI P_h, LLI P_w>
    static SPTR<Tensor<B * C * (P_w + W) * (P_h + H)>> PadInput(SPTR<Tensor<T>> x) requires (P_w > 1 && P_h == 1) {
        constexpr int leftSize = P_w / 2;
        constexpr int upSize = P_h / 2;

        constexpr int downSize = P_h - upSize;
        constexpr int rightSize = P_w - leftSize;

        constexpr int _upSize = upSize * B * C * (W + rightSize);
        constexpr int _leftSize = leftSize * B * C * (H + P_h);

        constexpr int _downSize = downSize * B * C * W;
        constexpr int _rightSize = rightSize * B * C * (H + downSize);

        auto down_pad = MiniTorch<downSize>::Zeros();
        down_pad = MiniTorch<downSize>::ReShape(down_pad, {1, 1, downSize, 1}); // {1, 1, DOWNSIZE, 1}
        auto repped_down_pad = MiniTorch<downSize>::template Repeat<downSize * B * C>(down_pad, {B, C, 1, 1});
        auto full_down_pad = MiniTorch<downSize * B * C>::template Repeat<downSize * B * C * W>(repped_down_pad, {1, 1, 1, W}); // {B, C, DOWNSIZE, W}

        auto right_pad = MiniTorch<rightSize>::Zeros();
        right_pad = MiniTorch<rightSize>::ReShape(right_pad, {1, 1, 1, rightSize}); // {1, 1, 1, RIGHTSIZE}
        auto repped_right_pad = MiniTorch<rightSize>::template Repeat<rightSize * B * C>(right_pad, {B, C, 1, 1});
        auto full_right_pad = MiniTorch<rightSize * B * C>::template Repeat<rightSize * B * C * (H + downSize)>(repped_right_pad, {1, 1, (H + downSize), 1}); // {B, C, DOWNSIZE, W}

        auto left_pad = MiniTorch<leftSize>::Zeros();
        left_pad = MiniTorch<leftSize>::ReShape(left_pad, {1, 1, 1, leftSize});
        auto repped_left_pad = MiniTorch<leftSize>::template Repeat<leftSize * B * C>(left_pad, {B, C, 1, 1});
        auto full_left_pad = MiniTorch<leftSize * B * C>::template Repeat<leftSize * B * C * (H + P_h)>(repped_left_pad, {1, 1, H + P_h, 1}); // {B, C, H, RIGHTSIZE}

        auto downed = MiniTorch<T>::template Concatenate<_downSize>(x, full_down_pad, 2);
        auto righted = MiniTorch<T + _downSize>::template Concatenate<_rightSize>(downed, full_right_pad, 3);
        auto lefted = MiniTorch<_leftSize>::template Concatenate<T + _downSize + _rightSize + _upSize>(full_left_pad, righted, 3);

        return lefted;
    }
    template<LLI B, LLI C, LLI H, LLI W, LLI P_h, LLI P_w>
    static SPTR<Tensor<B * C * (P_w + W) * (P_h + H)>> PadInput(SPTR<Tensor<T>> x) requires (P_w == 1 && P_h == 1) {
        constexpr int leftSize = P_w / 2;
        constexpr int upSize = P_h / 2;

        constexpr int downSize = P_h - upSize;
        constexpr int rightSize = P_w - leftSize;

        // constexpr int _upSize = upSize * B * C * (W + rightSize);
        // constexpr int _leftSize = leftSize * B * C * (H + P_h);

        constexpr int _downSize = downSize * B * C * W;
        constexpr int _rightSize = rightSize * B * C * (H + downSize);

        auto down_pad = MiniTorch<downSize>::Zeros();
        down_pad = MiniTorch<downSize>::ReShape(down_pad, {1, 1, downSize, 1}); // {1, 1, DOWNSIZE, 1}
        auto repped_down_pad = MiniTorch<downSize>::template Repeat<downSize * B * C>(down_pad, {B, C, 1, 1});
        auto full_down_pad = MiniTorch<downSize * B * C>::template Repeat<downSize * B * C * W>(repped_down_pad, {1, 1, 1, W}); // {B, C, DOWNSIZE, W}

        auto right_pad = MiniTorch<rightSize>::Zeros();
        right_pad = MiniTorch<rightSize>::ReShape(right_pad, {1, 1, 1, rightSize}); // {1, 1, 1, RIGHTSIZE}
        auto repped_right_pad = MiniTorch<rightSize>::template Repeat<rightSize * B * C>(right_pad, {B, C, 1, 1});
        auto full_right_pad = MiniTorch<rightSize * B * C>::template Repeat<rightSize * B * C * (H + downSize)>(repped_right_pad, {1, 1, (H + downSize), 1}); // {B, C, DOWNSIZE, W}

        auto downed = MiniTorch<T>::template Concatenate<_downSize>(x, full_down_pad, 2);
        auto righted = MiniTorch<T + _downSize>::template Concatenate<_rightSize>(downed, full_right_pad, 3);

        return righted;
    }

    template<LLI T2>
    static SPTR<Tensor<T2>> Gather(const SPTR<Tensor<T>> &t1, const int dim, const SPTR<Tensor<T2>> &indexes) {
        //A{3,5,5} - I{1,5,5}

        if (indexes->getShape().size() > t1->getShape().size()) {
            throw std::runtime_error("Bad Index!");
        }

        ARR<PTR_E, T2> out;
        for (int i = 0; i < T2; i++) {
            VEC_I index = CreateIndex<T2>(indexes, i);
            index[dim] = TakeObject(indexes, i)->getData();

            out[i] = TakeObject<T>(t1, index);
        }

        return std::make_shared<Tensor<T2>>(out, indexes->getShape());
    }

    // 2 2 2 - 2 1 2 - d1
    // 2 2 2 - 2 1 2 - d1 -TRANSPOSE> 2 2 2 - 1 2 2 - d0 -> 3 2 2 -TRANSPOSE> 2 3 2
    // 2 2 2 - 1 1 2 - d2 -TRANSPOSE> 2 2 2 - 2 1 1 - d0 -> 3 2 2 -TRANSPOSE> 2 3 2
    template<LLI T2>
    static SPTR<Tensor<T + T2>> Concatenate(const SPTR<Tensor<T>> &t1, const SPTR<Tensor<T2>> &t2, int dim) requires (T > T2) {
        const VEC_I &t1_shape = t1->getShape();
        const VEC_I &t2_shape = t2->getShape();

        if (dim < 0) {
            dim += t1->getShape().size();
        }

        if (dim >= t1_shape.size() || dim < 0) {
            throw std::runtime_error("Bad Dimension!");
        }

        if (t1_shape.size() != t2_shape.size()) {
            throw std::runtime_error("Incompatible Shapes!");
        }

        const SPTR<Tensor<T>> &ref_ = MiniTorch<T>::TakeTranspose(t1, dim, 0);
        const SPTR<Tensor<T2>> &ref2_ = MiniTorch<T2>::TakeTranspose(t2, dim, 0);

        VEC_I f_shape = ref_->getShape();
        f_shape[0] += ref2_->getShape()[0];

        const ARR<PTR_E, T> &ref = ref_->getData();
        const ARR<PTR_E, T2> &ref2 = ref2_->getData();

        const int t1_size = ref.size();
        const int t2_size = ref2.size();

        ARR<PTR_E, T + T2> out;
        for (int i = 0; i < t1_size; i++) {
            out[i] = ref[i];
        }
        for (int i = 0; i < t2_size; i++) {
            out[i + ref.size()] = ref2[i];
        }

        SPTR<Tensor<T + T2>> out_tensor = std::make_shared<Tensor<T + T2>>(out, f_shape);

        out_tensor = MiniTorch<T + T2>::TakeTranspose(out_tensor, dim, 0);

        return out_tensor;
    }
    template<LLI T2>
    static SPTR<Tensor<T + T2>> Concatenate(const SPTR<Tensor<T>> &t1, const SPTR<Tensor<T2>> &t2, int dim) {
        const VEC_I &t1_shape = t1->getShape();
        const VEC_I &t2_shape = t2->getShape();

        if (dim < 0) {
            dim += t1->getShape().size();
        }

        if (dim >= t1_shape.size() || dim < 0) {
            throw std::runtime_error("Bad Dimension!");
        }

        if (t1_shape.size() != t2_shape.size()) {
            throw std::runtime_error("Incompatible Shapes!");
        }

        const SPTR<Tensor<T>> &ref_ = MiniTorch<T>::TakeTranspose(t1, dim, 0);
        const SPTR<Tensor<T2>> &ref2_ = MiniTorch<T2>::TakeTranspose(t2, dim, 0);

        VEC_I f_shape = ref_->getShape();
        f_shape[0] += ref2_->getShape()[0];

        const ARR<PTR_E, T> &ref = ref_->getData();
        const ARR<PTR_E, T2> &ref2 = ref2_->getData();

        const int t1_size = ref.size();
        const int t2_size = ref2.size();

        ARR<PTR_E, T + T2> out;
        for (int i = 0; i < t1_size; i++) {
            out[i] = ref[i];
        }
        for (int i = 0; i < t2_size; i++) {
            out[i + ref.size()] = ref2[i];
        }

        SPTR<Tensor<T + T2>> out_tensor = std::make_shared<Tensor<T + T2>>(out, f_shape);

        out_tensor = MiniTorch<T + T2>::TakeTranspose(out_tensor, dim, 0);

        return out_tensor;
    }

    static SPTR<Tensor<T>> Where(const SPTR<Tensor<T>> &mask, DTYPE condition, DTYPE one, DTYPE zero) {
        ARR<PTR_E, T> out;
        for (int i = 0; i < mask->Numel(); i++) {
            const DTYPE temp_data = mask->getData()[i]->getData();

            if (temp_data == condition) {
                const PTR_E &temp = std::make_shared<Element>(one);
                out[i] = temp;
            }
            else {
                const PTR_E &temp = std::make_shared<Element>(zero);
                out[i] = temp;
            }
        }

        return std::make_shared<Tensor<T>>(out, mask->getShape());
    }

    static SPTR<Tensor<T>> Where(const SPTR<Tensor<T>> &mask, DTYPE one, DTYPE zero) {
        ARR<PTR_E, T> out;
        for (int i = 0; i < mask->Numel(); i++) {
            const DTYPE temp_data = mask->getData()[i]->getData();

            if (temp_data > 0) {
                const PTR_E &temp = std::make_shared<Element>(one);
                out[i] = temp;
            }
            else {
                const PTR_E &temp = std::make_shared<Element>(zero);
                out[i] = temp;
            }
        }

        return std::make_shared<Tensor<T>>(out, mask->getShape());
    }

    // static void ElementWiseOpt(const SPTR<Tensor<T>> &t1, DTYPE(*op)(DTYPE));

    template<LLI D, LLI C>
    static ARR<std::variant<SPTR<Tensor<T / D * ((D - D % C) / C)>>,
        SPTR<Tensor<(T / D) * (D % C)>>>, D / C + 1 * (D % C != 0)>
            Chunk(const SPTR<Tensor<T>> &t1, const int dim) requires(C > 0 && D > 0 && T > 0)
    {
        if (dim < 0) {
            dim += t1->getShape().size();
        }
        if (dim < 0) {
            throw std::runtime_error("Bad Dimension!");
        }
        if (C >= t1->getShape()[dim]) {
            ARR<SPTR<Tensor<T>>, 1> temp{t1};

            return temp;
        }

        if (dim != 0) {
            SPTR<Tensor<T>> transposed = TakeTranspose(t1, dim, 0);

            auto out = Chunk<D, C>(transposed, 0);

            for(int i = 0; i < out.size(); ++i) {
                if (i != out.size() - 1){
                    out[i] = MiniTorch<(T / D) * ((D - D % C) / C)>::TakeTranspose
                    (std::get<SPTR<Tensor<T / D * ((D - D % C) / C)>>>(out[i]), dim, 0);
                }
                else {
                    out[i] = MiniTorch<(T / D) * (D % C)>::TakeTranspose(std::get<SPTR<Tensor<(T / D) * (D % C)>>>(out[i]), dim, 0);
                }
            }

            return out;
        }

        const ARR<PTR_E, T> &ref = t1->getData();

        constexpr bool perfect = D % C == 0;
        constexpr int n_iteration = 1 + D / C;

        constexpr int chunked_dim_size = D / C;
        constexpr int singular_number_elements = T / D; // 2 1 2 toplam eleman sayisi;
        constexpr int number_elements = singular_number_elements * chunked_dim_size; // -> singular ile toplam dimi carp normaller icin 2 1 2'den 4 tane var mesela;
        constexpr int last_number_elements = singular_number_elements * (D % C); // singularla sayiyi carp 2 1 2'den 1 tane kaldi;

        VEC_I full_shape = t1->getShape();
        full_shape[0] /= C;

        VEC_I last_shape = t1->getShape();
        last_shape[0] %= C;

        ARR<std::variant<SPTR<Tensor<number_elements>>, SPTR<Tensor<last_number_elements>>>, n_iteration> out{};
        for (int i = 0; i < n_iteration; ++i) {
            if (i != n_iteration - 1 || perfect) {
                ARR<PTR_E, number_elements> temp_data;

                for (int e = 0; e < number_elements; ++e) {
                    temp_data[e] = ref[i * number_elements + e]->Copy();
                }

                out[i] = std::make_shared<Tensor<number_elements>>(temp_data, full_shape);
            }
            else {
                ARR<PTR_E, last_number_elements> temp_data;

                for (int e = 0; e < last_number_elements; ++e) {
                    temp_data[e] = ref[i * number_elements + e]->Copy();
                }

                out[i] = std::make_shared<Tensor<last_number_elements>>(temp_data, last_shape);
            }
        }

        return out;
    }

    template<LLI N>
    static SPTR<Tensor<T * T>> Vander(const SPTR<Tensor<T>> &t1) {
        const ARR<PTR_E, T> &ref = t1->getData();

        ARR<PTR_E, T * T> out;
        for (int p = N > 0 ? N : T; p > 0; --p) {
            for (int i = 0; i < T; ++i) {
                PTR_E temp = std::make_shared<Element>(std::pow(ref[i]->getData(), p-1));
                out[T * (T - p) + i] = temp;
            }
        }

        VEC_I shape{N > 0 ? N : T, T};

        SPTR<Tensor<T * T>> temp = std::make_shared<Tensor<T * T>>(out, shape);
        temp = MiniTorch<T * T>::TakeTranspose(temp, 0, 1);

        return temp;
    }

    static SPTR<Tensor<T>> LinSpace(const double start, const double end) {
        if (T < 1) {
            throw std::runtime_error("Bad Length!");
        }
        ARR<PTR_E, T> out;

        const double hop_size = (end - start) / T;
        for (int i = 0; i < T; i++) {
            const PTR_E &temp = std::make_shared<Element>(start + i * hop_size);
            out[i] = temp;
        }

        VEC_I shape;
        shape.reserve(1);

        shape.push_back(out.size());

        return std::make_shared<Tensor<T>>(out, shape);
    }

    static SPTR<Tensor<T>> LogSpace(const double start, const double end) {
        if (T < 1) {
            throw std::runtime_error("Bad Length!");
        }
        ARR<PTR_E, T> out;

        const int s = std::pow(2, start);
        const double hop_size = (std::pow(2, end) - s) / T;

        for (int i = 0; i < T; i++) {
            const PTR_E &temp = std::make_shared<Element>(s + i * hop_size);
            out[i] = temp;
        }

        VEC_I shape{T};

        return std::make_shared<Tensor<T>>(out, shape);
    }

    static SPTR<Tensor<T>> Zeros() {
        if (T < 1) {
            throw std::runtime_error("Bad Length!");
        }
        ARR<PTR_E, T> out;

        for (int i = 0; i < T; i++) {
            const PTR_E &temp = std::make_shared<Element>(.0);
            out[i] = temp;
        }

        SPTR<Tensor<T>> t = std::make_shared<Tensor<T>>(out);

        return t;
    }

    static SPTR<Tensor<T>> Ones() {
        if (T < 1) {
            throw std::runtime_error("Bad Length!");
        }
        ARR<PTR_E, T> out;

        for (int i = 0; i < T; i++) {
            const PTR_E &temp = std::make_shared<Element>(1.0);
            out[i] = temp;
        }

        return std::make_shared<Tensor<T>>(out);
    }

    template<LLI T2=0, LLI T3>
    static SPTR<Tensor<(T - T2) / T3>> Arange() requires(T >= T2 && (T - T2) % T3 == 0) {
        constexpr int start = T2;
        constexpr int end = T;

        ARR<PTR_E, (T - T2) / T3> out;

        for (int i = start; i < end; i += T3) {
            out[(i - start) / T3] = std::make_shared<Element>(i);
        }

        VEC_I shape;
        shape.reserve(1);

        shape.push_back(out.size());

        return std::make_shared<Tensor<(T - T2) / T3>>(out, shape);
    }

    template<LLI T2=0>
    static SPTR<Tensor<T - T2>> Arange() requires(T >= T2) {
        constexpr int start = T2;
        constexpr int end = T;

        ARR<PTR_E, T - T2> out;

        for (int i = start; i < end; ++i) {
            out[i - start] = std::make_shared<Element>(i);
        }

        VEC_I shape;
        shape.reserve(1);

        shape.push_back(out.size());

        return std::make_shared<Tensor<T - T2>>(out, shape);
    }

    template<LLI T2>
    static SPTR<Tensor<T2 - T>> Arange() requires(T2 > T) {
        constexpr int start = T;
        constexpr int end = T2;

        ARR<PTR_E, T2 - T> out;

        for (int i = start; i < end; ++i) {
            const PTR_E &temp = std::make_shared<Element>(i);
            out[i - start] = temp;
        }

        VEC_I shape;
        shape.reserve(1);

        shape.push_back(out.size());

        return std::make_shared<Tensor<T2 - T>>(out, shape);
    }

    static SPTR<Tensor<T>> Diagonal() {
        if (T < 1) {
            throw std::runtime_error("Bad Length!");
        }
        ARR<PTR_E, T * T> outer;

        for (int r = 0; r < T; r++) {
            for (int c = 0; c < T; c++) {
                if (r == c) {
                    const PTR_E &temp = std::make_shared<Element>(1.0);
                    outer[r * T + c] = temp;
                }
                else {
                    const PTR_E &temp = std::make_shared<Element>(.0);
                    outer[r * T + c] = temp;
                }
            }
        }

        VEC_I shape = {T, T};

        return std::make_shared<Tensor<T * T>>(outer, shape);
    }

    static SPTR<Tensor<T * T>> Tril() {
        if (T < 1) {
            throw std::runtime_error("Bad Length!");
        }
        ARR<PTR_E, T * T> out;

        int f = 0;
        for (int r = 1; r <= T; r++) {
            for (int c = 0; c < r; c++) {
                out[f] = std::make_shared<Element>(1.0);
                ++f;
            }
            for (int c = 0; c < T - r; c++) {
                out[f] = std::make_shared<Element>(.0);
                ++f;
            }
        }

        VEC_I shape{T, T};

        return std::make_shared<Tensor<T * T>>(out, shape);
    }

    static SPTR<Tensor<T>> CreateOneHot(int index, const bool row=true) {
        if (T < 1) {
            throw std::runtime_error("Bad Length!");
        }
        if (index >= T) {
            throw std::runtime_error("Bad Index!");
        }
        if (index < 0) {
            index += T;
        }
        if (index < 0) {
            throw std::runtime_error("Bad Index!");
        }

        auto t1 = MiniTorch<T>::Zeros();

        VEC_I shape;
        shape.reserve(3);

        if (row) {
            shape = {1, 1, T};
        }else {
            shape = {1, T, 1};
        }

        auto one = std::make_shared<Element>(1.0);

        t1 = ReShape(t1, shape);
        t1->SetObject({index}, one);

        return t1;
    }

    // %% INITIALIZATION %%
    static SPTR<Tensor<T>> HeUniformTensorInitialization(const int f_in, VEC_I shape = {}) {
        if (T < 1) {
            throw std::runtime_error("Bad Length!");
        }
        if (f_in < 1) {
            throw std::runtime_error("Bad F_IN!");
        }
        static std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<DTYPE> dist(-std::sqrt(6.0 / f_in), std::sqrt(6.0 / f_in));

        ARR<PTR_E, T> elements;
        for (int i = 0; i < T; i++) {
            const DTYPE &u1 = dist(rng);

            elements[i] = std::make_shared<Element>(u1);
        }

        if (shape.size() > 0) {
            return std::make_shared<Tensor<T>>(elements, shape, false, true);
        }

        return std::make_shared<Tensor<T>>(elements, false, true);
    }
};
