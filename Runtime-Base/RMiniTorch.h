#pragma once

class RMiniTorch {
    // %% UTILITY FUNCS %%
    static VEC_S SplitByChar(const String &data, const char special){
        VEC_S out;

        int s = 0;
        for (int i = 0; i < data.size(); i++) {
            if (data[i] == special) {
                out.push_back(data.substr(s, i - s));
                s = i + 1;
            }
        }

        return out;
    }

    static VEC_I CalcSliceShape(const VEC_I &start, const VEC_I &end) {
        VEC_I shape{};

        for (std::size_t i = 0; i < start.size(); ++i)
            shape.push_back(end[i] - start[i] + 1);

        return shape;
    }

    static int CalcSliceSize(const VEC_I &start, const VEC_I &end)
    {
        int w = 1;
        for (int i = 0; i < 2; ++i) {
            w *= end[i] - start[i] + 1;
        }

        return w;
    }

    static VEC_I CreateIndex(const PTR_T &t1, const int n) {
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

    static VEC_I TransformShape(const PTR_T &t1, const VEC_I &shape) {
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

    static VEC_I TransformIndex(const PTR_T &t1, const VEC_I &index) {
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
    static int TransformIndex(const PTR_T &t1, const int &index) {
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
    ~RMiniTorch() = delete;
    RMiniTorch() = delete;
    RMiniTorch(const RMiniTorch&) = delete;
    RMiniTorch(RMiniTorch&&) = delete;
    RMiniTorch& operator=(const RMiniTorch&) = delete;
    RMiniTorch& operator=(RMiniTorch&&) = delete;

    // %% SET TENSORS %%
    static void ZeroStartData(const PTR_T &t1) {
        const auto &ref = t1->getData();

        for (int i = 0; i < ref.size(); ++i) {
            auto temp_e = std::make_shared<Element>(.0);
            t1->SetObject(i, temp_e);
        }
    }

    static void PartialStartData(const PTR_T &t1, const PTR_T &t2) {
        const auto &ref = t1->getData();

        for (int i = 0; i < ref.size(); ++i) {
            t2->SetObject(i, ref[i]);
        }
    }

    static void PartialIndexStartData(const PTR_T &t1, const PTR_T &t2) {
        const auto &ref = t1->getData();

        for (int i = 0; i < ref.size(); ++i) {
            auto ref_i = Minitorch::CreateIndex(t1, i);
            t2->SetObject(ref_i, ref[i]);
        }
    }

    // // %% TENSOR %%
    static PTR_T TakeSlice(const PTR_T &t1, const VEC_I &start, const VEC_I &end) {
        const VEC_I &start_indexes = TransformIndex(t1, start);
        const VEC_I &end_indexes = TransformIndex(t1, end);

        const VEC_E &ref = t1->getData();

        VEC_I n_shape = CalcSliceShape(start_indexes, end_indexes);

        VEC_E elements;

        int add_index = 0;
        for (int n = 0; n < ref.size(); n++) {
            VEC_I cor_indexing = CreateIndex(t1, n);

            bool add = true;
            for (int t = 0; t < start_indexes.size(); t++) {
                if (cor_indexing[t] > end_indexes[t] || cor_indexing[t] < start_indexes[t]) {
                    add = false;
                    break;
                }
            }
            if (add) { elements.push_back(ref[n]->Copy()); ++add_index; }
        }

        auto t_out = std::make_shared<Tensor>(elements);
        t_out = Minitorch::ReShape(t_out, n_shape);

        return t_out;
    }

    static PTR_E TakeObject(const PTR_T &t1, const VEC_I &indexes) {
        const VEC_I &indexing = TransformIndex(t1, indexes);

        const VEC_E &ref = t1->getData();
        const VEC_I &ref2 = t1->getIndexWeights();

        int index = 0;
        for (int i = 0; i < indexing.size(); i++) {
            index += indexing[i] * ref2[i];
        }

        return ref[index];
    }
    static PTR_E TakeObject(const PTR_T &t1, VEC_I &&indexes) {
        const VEC_I &indexing = TransformIndex(t1, indexes);

        const VEC_E &ref = t1->getData();
        const VEC_I &ref2 = t1->getIndexWeights();

        int index = 0;
        for (int i = 0; i < indexing.size(); i++) {
            index += indexing[i] * ref2[i];
        }

        return ref[index];
    }
    static PTR_E TakeObject(const PTR_T &t1, const int indexes) {
        const int &indexing = TransformIndex(t1, indexes);

        const VEC_E &ref = t1->getData();

        return ref[indexing];
    }

    static PTR_T ReShape(const PTR_T &t1, const VEC_I &shape) {
        const VEC_I &new_shape = TransformShape(t1, shape);

        const PTR_T &temp = t1->Copy();

        temp->setShape(new_shape);

        return temp;
    }

    static PTR_T Repeat(const PTR_T &t1, const VEC_I &repeats) {
        NonZero(repeats);

        if (t1->getShape().size() != repeats.size()) {
            throw std::runtime_error("Bad Repeat!");
        }

        PTR_T out = t1;
        for (int rep_dim = repeats.size() - 1; rep_dim >= 0; --rep_dim) {
            for (int rep = 1; rep < repeats[rep_dim];) {
                if (rep * 2 <= repeats[rep_dim]) {
                    out = Concatenate(out, out, rep_dim);
                    rep *= 2;
                }
                else {
                    out = Concatenate(out, t1, rep_dim);
                    ++rep;
                }
            }
        }

        return out;
    }

    static PTR_T Clamp(const PTR_T &t1, DTYPE min, DTYPE max) {
        const VEC_E &ref = t1->getData();

        VEC_E out;
        for (const auto & i : ref) {
            if (i->getData() > max) {
                out.push_back(std::make_shared<Element>(max, .0, .0));
            }
            else if (i->getData() < min) {
                out.push_back(std::make_shared<Element>(min, .0, .0));
            }
            else {
                out.push_back(std::make_shared<Element>(i->getData(), .0, i->getBackScalar0()));
            }
        }

        return std::make_shared<Tensor>(out, t1->getShape());
    }

    static PTR_T TakeTranspose(const PTR_T &t1, const int dim_a, const int dim_b) {
        const VEC_E &ref = t1->getData();

        PTR_T out = t1->Copy();

        VEC_I n_shape = out->getShape();

        int temp = n_shape[dim_a];
        n_shape[dim_a] = n_shape[dim_b];
        n_shape[dim_b] = temp;

        out = ReShape(out, n_shape);

        for (int i = 0; i < ref.size(); i++) {
            VEC_I cor_indexing = CreateIndex(t1, i);

            temp = cor_indexing[dim_a];
            cor_indexing[dim_a] = cor_indexing[dim_b];
            cor_indexing[dim_b] = temp;

            out->SetObject(cor_indexing, ref[i]->Copy());
        }

        return out;
    }

    static PTR_T UnSqueeze(const PTR_T &t1, const int dim) {
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

        PTR_T out_tensor = t1->Copy();
        out_tensor = ReShape(out_tensor, good_shape);

        return out_tensor;
    }

    static PTR_T Squeeze(const PTR_T &t1, const int dim) {
        VEC_I good_shape;

        for (int i=0; i < t1->getShape().size(); i++) {
            if (i == dim && t1->getShape()[i] < 2) {
                continue;
            }
            good_shape.push_back(t1->getShape()[i]);
        }

        const PTR_T &temp = t1->Copy();

        temp->setShape(good_shape);

        return temp;
    }

    static PTR_T ViewAsComplex(const PTR_T &t1) {
        const VEC_E &ref = t1->getData();

        if (t1->getComplex()) {
            return t1->Copy();
        }

        VEC_I shape = t1->getShape();
        shape[shape.size()] /= 2; // (3,4,10) -> (3,4,5)
        shape.push_back(2); // (3,4,5) -> (3,4,5,2)

        VEC_E out;
        for (int i = 0; i < t1->getData().size(); i ++) {
            PTR_E temp = std::make_shared<Element>(ref[i]->getData());
            out.push_back(temp);

            PTR_E tempim = std::make_shared<Element>(ref[i+1]->getImData());
            out.push_back(tempim);
        }

        return std::make_shared<Tensor>(out, shape, true);
    }

    static PTR_T ViewAsReal(const PTR_T &t1) {
        const VEC_E &ref = t1->getData();

        if (!t1->getComplex()) {
            return t1->Copy();
        }

        VEC_I shape = t1->getShape();
        shape.pop_back(); // (3,4,5,2) -> (3,4,5)
        shape[shape.size()] *= 2; // (3,4,5) -> (3,4,10)

        VEC_E out;
        for (int i = 0; i < t1->getData().size(); ++++i) {
            PTR_E temp = std::make_shared<Element>(ref[i]->getData(), .0);
            PTR_E temp2 = std::make_shared<Element>(ref[i]->getImData(), .0);
            out.push_back(temp);
            out.push_back(temp2);
        }

        return std::make_shared<Tensor>(out, shape, false);
    }

    static VEC_D AsVector(const PTR_T &t1){
        const VEC_E &ref = t1->getData();

        VEC_D out;
        for (const auto & i : ref) {
            out.push_back(i->getData());
        }

        return out;
    }

    static PTR_E Amax(const PTR_T &t1) {
        const VEC_E &ref = t1->getData();

        PTR_E amax = t1->getData()[0]->Copy();
        for (int i = 1; i < ref.size(); i++) {
            if (ref[i]->getData() > amax->getData()) {
                amax = ref[i];
            }
        }

        return amax;
    }

    static PTR_E Amin(const PTR_T &t1) {
        const VEC_E &ref = t1->getData();

        PTR_E amin = ref[0]->Copy();
        for (int i = 1; i < ref.size(); i++) {
            if (ref[i]->getData() < amin->getData()) {
                amin = ref[i];
            }
        }

        return amin;
    }
    
    static PTR_T Sum(const PTR_T &t1, const int dim) {
        PTR_T temp_t1;
        if (dim != 0) {
            temp_t1 = TakeTranspose(t1, dim, 0);
        }
        else {
            temp_t1 = t1;
        }

        const int T = temp_t1->getData().size();
        const int T2 = temp_t1->getShape()[0];

        VEC_I new_shape = temp_t1->getShape();
        new_shape[0] = 1;

        PTR_T out = Zeros(T / T2);
        out = ReShape(out, new_shape);

        // 3 4 5 -1d == 4 3 5 - 0d
        for (int d = 0; d < (T/T2); ++d) {
            PTR_E temp_element = std::make_shared<Element>(.0);
            for (int i = 0; i < T2; ++i) {
                PTR_E take_element = TakeObject(temp_t1, i * temp_t1->getIndexWeights()[0] + d);
                temp_element = AddElement(temp_element, take_element);
            }

            out->SetObject(d, temp_element);
        }

        out = TakeTranspose(out, dim, 0);

        return out;
    }

    static PTR_E Sum(const PTR_T &t1, const int T2, const int offset) {
        const auto &ref = t1->getData();

        auto out = std::make_shared<Element>(.0);
        for (int i = offset; i < T2 + offset; ++i) {
            out = AddElement(out, ref[i]);
        }

        return out;
    }
    
    static PTR_E Sum(const PTR_T &t1) {
        const VEC_E &ref = t1->getData();

        auto out = std::make_shared<Element>(.0);

        for (const PTR_E& i : ref) {
            out = AddElement(out, i);
        }

        return out;
    }

    static PTR_E Mean(const PTR_T &t1, const int T2, const int offset) {
        auto out = Sum(t1, T2, offset);

        out = Minitorch::MulElement(out, 1.0l / T2);

        return out;
    }
    
    static PTR_E Mean(const PTR_T &t1) {
        const VEC_E &ref = t1->getData();

        auto out = Sum(t1);

        out = MulElement(out, 1.0l/ref.size());

        return out;
    }
    
    static PTR_E Var(const PTR_T &t1, const int T2, const int offset) {
        const auto &ref = t1->getData();

        const auto mean = Mean(t1, T2, offset);
        const auto neg_mean = MulElement(mean, -1);

        auto out = std::make_shared<Element>(.0);
        for (int i = offset; i < T2 + offset; ++i) {
            const PTR_E &temp = ref[i];
            PTR_E copied = temp->Copy();

            copied = Minitorch::AddElement(copied, neg_mean);
            copied = Minitorch::Pow(copied, 2);

            out = Minitorch::AddElement(out, copied);
        }
        out = Minitorch::MulElement(out, 1.0l/(T2 - 1));

        return out;
    }

    static PTR_E Var(const PTR_T &t1) {
        const VEC_E &ref = t1->getData();

        const auto mean = Mean(t1);

        auto out = std::make_shared<Element>(.0);
        for (const auto & temp : ref) {
            PTR_E copied = temp->Copy();

            copied = MulElement(AddElement(MulElement(copied, -1), mean),-1);
            copied = Pow(copied, 2);

            out = AddElement(out, copied);
        }
        out = MulElement(out, 1.0l/ref.size());

        return out;
    }
    
    static PTR_E Std(const PTR_T &t1, const int T2, const int offset) {
        return Minitorch::Pow(Var(t1, T2, offset), 0.5);
    }

    static PTR_E Std(const PTR_T &t1) {
        return Pow(Var(t1), 0.5);
    }
    
    static PTR_E RMS(const PTR_T &t1, const int T2, const int offset) {
        const auto &ref = t1->getData();

        auto out = std::make_shared<Element>(.0);
        for (int i = offset; i < T2 + offset; ++i) {
            auto copied = Minitorch::Pow(ref[i]->Copy(), 2);

            out = Minitorch::AddElement(out, copied);
        }
        out = Minitorch::MulElement(out, 1.0l / T2);
        out = Minitorch::Pow(out, 0.5);

        return out;
    }

    static PTR_E RMS(const PTR_T &t1) {
        const auto &ref = t1->getData();

        auto out = std::make_shared<Element>(.0);
        for (int i = 0; i < ref.size(); ++i) {
            auto copied = Minitorch::Pow(ref[i]->Copy(), 2);

            out = Minitorch::AddElement(out, copied);
        }
        out = Minitorch::MulElement(out, 1.0l / ref.size());
        out = Minitorch::Pow(out, 0.5);

        return out;
    }

    static VEC_I Argmax(const PTR_T &t1) {
        const VEC_E &ref = t1->getData();

        int index = 0;
        for (int i = 1; i < ref.size(); i++) {
            if (ref[i]->getData() > ref[index]->getData()) {
                index = i;
            }
        }

        const VEC_I &out = CreateIndex(t1, index);

        return out;
    }

    static VEC_I Argmin(const PTR_T &t1) {
        const VEC_E &ref = t1->getData();

        int index = 0;
        for (int i = 1; i < ref.size(); i++) {
            if (ref[i]->getData() < ref[index]->getData()) {
                index = i;
            }
        }

        const VEC_I &out = CreateIndex(t1, index);

        return out;
    }

    static void Printensor(const PTR_T &t1) {
        const VEC_E &ref = t1->getData();

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

            if ((i+1) % (shape[shape.size()-1]) == 0) { cout << TakeObject(t1, i)->getData(); }
            else { cout << TakeObject(t1, i)->getData() << " "; }

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

    // %% ELEMEN OP. FUNCIONS %%
    // t1.shape.size >= 4 && t2.shape.size >= 4
    static PTR_T MatMul(const PTR_T &t1, const PTR_T &t2) {
        // [c, a, m] x [c, m, b] -> [c, a,b]
        // [2, 3, 4] x [1, 4, 3] -> Calismali
        // [2, 3, 4] x [5, 4, 3] -> Hata
        // artik 4lu
        const VEC_I &t1_shape = t1->getShape();
        const VEC_I &t2_shape = t2->getShape();

        if (t1_shape[t1_shape.size() - 1] != t2_shape[t2_shape.size() - 2]) {
            throw std::runtime_error("Bad Tensor Shapes!");
        }

        VEC_I a_shape = {-1, t1_shape[t1_shape.size() - 2], t1_shape[t1_shape.size() - 1]};
        VEC_I b_shape = {-1, t2_shape[t2_shape.size() - 2], t2_shape[t2_shape.size() - 1]};

        const PTR_T &temp_t1 = ReShape(t1, a_shape);
        const PTR_T &temp_t2 = ReShape(t2, b_shape);

        a_shape = temp_t1->getShape();
        b_shape = temp_t2->getShape();

        VEC_I matted_shape = {t1_shape[t1_shape.size() - 4], t1_shape[t1_shape.size() - 3], a_shape[1], b_shape[2]};

        bool use_c = a_shape[0] == b_shape[0];

        // [1 2] [7 8] -> [n m]
        // [3 4] [5 6] -> [a b]
        VEC_E out;
        for (int tdim = 0; tdim < a_shape[0]; ++tdim){
            const int mul_count = a_shape[a_shape.size() - 1];

            for (int r = 0; r < a_shape[1]; ++r) {
                for (int c = 0; c < b_shape[2]; ++c) {
                    PTR_E s = std::make_shared<Element>(.0);

                    for (int i = 0; i < mul_count; i++) {
                        s = AddElement(s, MulElement(TakeObject(temp_t1, {tdim, r, i}), TakeObject(temp_t2, {use_c ? tdim : 0, i, c})));
                    }

                    out.push_back(s);
                }
            }
        }

        auto t_out = std::make_shared<Tensor>(out, matted_shape);

        if (t1_shape.size() == 2) {
            t_out = Squeeze(t_out, 0);
        }

        return t_out;
    }

    static PTR_T FlexibleMul(const PTR_T &t1, const PTR_T &t2, const VEC_I start, const VEC_I end) {
        // T1: (B, N, M) | T2: (B, 1, M) - (1, N, M) - (B, N, 1) - (B, 1, 1) - (1, N, 1) - (1, 1, 1) ALL VALID
        VEC_E out;

        int t2_counter = 0;
        for (int i = 0; i < t1->getData().size(); ++i) {
            t2_counter %= t2->getData().size();

            VEC_I index = CreateIndex(t1, i); // -> (a, b, c)

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

            const PTR_E &a_element = TakeObject(t1, i);
            const PTR_E &b_element = TakeObject(t2, t2_counter);

            out.push_back(MulElement(a_element, b_element));

            ++t2_counter;
        }

        VEC_I new_shape = t2->getShape();
        new_shape[0] = -1;

        auto t5 = std::make_shared<Tensor>(out);
        t5 = Minitorch::ReShape(t5, new_shape);

        return t5;
    }
    static PTR_T FlexibleMul(const PTR_T &t1, const PTR_T &t2) {
        // 1: (B, N, M) | 2: (B, 1, M) - (1, N, M) - (B, N, 1) - (B, 1, 1) - (1, N, 1) - (1, 1, 1) ALL VALID

        PTR_T big_tensor;
        PTR_T small_tensor;
        if (t1->getData().size() >= t2->getData().size()) {
            big_tensor = t1;
            small_tensor = t2;
        }
        else {
            big_tensor = t2;
            small_tensor = t1;
        }

        const VEC_E &ref = big_tensor->getData();
        // const VEC_E &ref2 = t2->getData();

        VEC_E out;

        const VEC_I &a_shape = big_tensor->getShape();
        const VEC_I &b_shape = small_tensor->getShape();

        for (int i = 0; i < ref.size(); i++) {
            VEC_I index = CreateIndex(big_tensor, i); // -> (a, b, c)

            VEC_I free_index;
            free_index.reserve(index.size());
            for (int t = 0; t < index.size(); t++) {
                if (a_shape[t] == b_shape[t]) { free_index.push_back(index[t]); }
                else { free_index.push_back(0); }
            } // (a, b, c) -> if its free take "a" but if dim=1 is not free take "0" -> (1, b, c)

            const PTR_E &a_element = TakeObject(big_tensor, index);
            const PTR_E &b_element = TakeObject(small_tensor, free_index);

            out.push_back(MulElement(a_element, b_element));
        }

        return std::make_shared<Tensor>(out, a_shape);
    }
    static PTR_T FlexibleMul(const PTR_T &t1, const DTYPE b) {
        const VEC_E &ref = t1->getData();

        VEC_E out;
        for (const auto & i : ref) {
            out.push_back(MulElement(i, b));
        }

        return std::make_shared<Tensor>(out, t1->getShape());
    }
    static PTR_T FlexibleMul(const DTYPE b, const PTR_T &t1) {
        const VEC_E &ref = t1->getData();

        VEC_E out;
        for (const auto & i : ref) {
            out.push_back(MulElement(i, b));
        }

        return std::make_shared<Tensor>(out, t1->getShape());
    }

    static PTR_T FlexibleDiv(const PTR_T &t1, const PTR_T &t2){
        const VEC_E &ref = t1->getData();

        VEC_E out;

        const VEC_I &a_shape = t1->getShape();
        const VEC_I &b_shape = t2->getShape();

        for (int i = 0; i < ref.size(); i++) {
            VEC_I index = CreateIndex(t1, i); // -> (a, b, c)

            VEC_I free_index;
            free_index.reserve(index.size());
            for (int t = 0; t < index.size(); t++) {
                if (a_shape[t] == b_shape[t]) { free_index.push_back(index[t]); }
                else { free_index.push_back(0); }
            } // (a, b, c) -> if its free take "a" but if dim=1 is not free take "0" -> (1, b, c)

            const PTR_E &a_element = TakeObject(t1, index);
            const PTR_E &b_element = TakeObject(t2, free_index);

            out.push_back(DivElement(a_element, b_element));
        }

        return std::make_shared<Tensor>(out, a_shape);
    }
    static PTR_T FlexibleDiv(const PTR_T &t1, const DTYPE b) {
        const VEC_E &ref = t1->getData();

        VEC_E out;
        for (const auto & i : ref) {
            out.push_back(DivElement(i, b));
        }

        return std::make_shared<Tensor>(out, t1->getShape());
    }
    static PTR_T FlexibleDiv(const DTYPE b, const PTR_T &t1) {
        const VEC_E &ref = t1->getData();

        VEC_E out;
        for (const auto & i : ref) {
            out.push_back(DivElement(b, i));
        }

        return std::make_shared<Tensor>(out, t1->getShape());
    }

    static PTR_T FlexibleAdd(const PTR_T &t1, const PTR_T &t2) {
        PTR_T big_tensor;
        PTR_T small_tensor;
        if (t1->getData().size() >= t2->getData().size()) {
            big_tensor = t1;
            small_tensor = t2;
        }
        else {
            big_tensor = t2;
            small_tensor = t1;
        }

        const VEC_E &ref = big_tensor->getData();

        VEC_E out;

        const VEC_I &a_shape = big_tensor->getShape();
        VEC_I b_shape = small_tensor->getShape();

        while (b_shape.size() < a_shape.size()) {
            b_shape.insert(b_shape.begin(), 1);
        }
        small_tensor = ReShape(t2, b_shape);

        std::vector<bool> free_dims;
        for (int i = 0; i < a_shape.size(); i++) { free_dims.push_back(a_shape[i] == b_shape[i]); }

        for (int i = 0; i < ref.size(); i++) {
            VEC_I index = CreateIndex(big_tensor, i); // -> (a, b, c)

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

            const PTR_E &a_element = TakeObject(big_tensor, index);
            const PTR_E &b_element = TakeObject(small_tensor, free_index);

            out.push_back(AddElement(a_element, b_element));
        }

        return std::make_shared<Tensor>(out, a_shape, big_tensor->getIndexWeights(), big_tensor->getComplex());
    }
    static PTR_T FlexibleAdd(const PTR_T &t1, const DTYPE b) {
        const VEC_E &ref = t1->getData();

        VEC_E out;
        for (const auto & i : ref) {
            out.push_back(AddElement(i, b));
        }

        return std::make_shared<Tensor>(out, t1->getShape(), t1->getIndexWeights(), t1->getComplex());
    }
    static PTR_T FlexibleAdd(const DTYPE b, const PTR_T &t1) {
        const VEC_E &ref = t1->getData();

        VEC_E out;
        for (const auto & i : ref) {
            out.push_back(AddElement(i, b));
        }

        return std::make_shared<Tensor>(out, t1->getShape(), t1->getIndexWeights(), t1->getComplex());
    }

    static PTR_T FlexibleSub(const PTR_T &t1, const PTR_T &t2) {
        const VEC_E &ref = t1->getData();

        VEC_E out;

        const VEC_I &a_shape = t1->getShape();
        const VEC_I &b_shape = t2->getShape();

        std::vector<bool> free_dims;
        for (int i = 0; i < a_shape.size(); i++) { free_dims.push_back(a_shape[i] == b_shape[i]); }

        for (int i = 0; i < ref.size(); i++) {
            VEC_I index = CreateIndex(t1, i); // -> (a, b, c)

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

            const PTR_E &a_element = TakeObject(t1, index);
            const PTR_E &b_element = TakeObject(t2, free_index);

            out.push_back(SubElement(a_element, b_element));
        }

        return std::make_shared<Tensor>(out, a_shape, t1->getIndexWeights(), t1->getComplex());
    }
    static PTR_T FlexibleSub(const PTR_T &t1, const DTYPE b) {
        const VEC_E &ref = t1->getData();

        VEC_E out;
        for (const auto & i : ref) {
            out.push_back(SubElement(i, b));
        }

        return std::make_shared<Tensor>(out, t1->getShape());
    }
    static PTR_T FlexibleSub(const DTYPE b, const PTR_T &t1) {
        const VEC_E &ref = t1->getData();

        VEC_E out;
        for (const auto & i : ref) {
            out.push_back(SubElement(b, i));
        }

        return std::make_shared<Tensor>(out, t1->getShape());
    }

    static PTR_E MulElement(const PTR_E &e1, const PTR_E &e2) {
        return std::make_shared<Element>(e1->getData() * e2->getData(), e1, e2->getData(), e2, e1->getData());
    }
    static PTR_E MulElement(const PTR_E &e1, DTYPE scalar) {
        return std::make_shared<Element>(e1->getData() * scalar, e1, scalar);
    }

    static PTR_E DivElement(const PTR_E &e1, const PTR_E &e2) {
        return std::make_shared<Element>(e1->getData() / e2->getData(), e1, 1/e2->getData(), e2, -e1->getData()/std::pow(e2->getData(), 2));
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

    static PTR_T Pow(const PTR_T &t1, const DTYPE scalar) {
        const VEC_E &ref = t1->getData();

        const PTR_T &temp = std::make_shared<Tensor>();
        for (int i = 0; i < ref.size(); i++) {
            temp->getData().push_back(Pow(ref[i], scalar));
        }

        temp->setShape(t1->getShape());

        return temp;
    }
    static PTR_E Pow(const PTR_E &e1, const DTYPE scalar) {
        return std::make_shared<Element>(std::pow(e1->getData(), scalar), e1, scalar * pow(e1->getData(), scalar - 1));
    }

    static PTR_T Square(const PTR_T &t1){ return Pow(t1, 2); }
    static PTR_E Square(const PTR_E &e1){ return Pow(e1, 2); }

    static PTR_T Sqrt(const PTR_T &t1) { return Pow(t1, 0.5); }
    static PTR_E Sqrt(const PTR_E &e1) { return Pow(e1, 0.5); }

    static PTR_T Abs(const PTR_T &t1) {
        const VEC_E &ref = t1->getData();

        const PTR_T &temp = std::make_shared<Tensor>();
        for (int i = 0; i < ref.size(); i++) {
            temp->getData().push_back(Abs(ref[i]));
        }

        temp->setShape(t1->getShape());

        return temp;
    }
    static PTR_E Abs(const PTR_E &e1) {
        if (e1->getData() < 0) {
            return std::make_shared<Element>(e1->getData() * -1, e1->getImData(), .0, e1->Copy(), -1);
        }
        return std::make_shared<Element>(e1->getData(), e1->getImData(), .0, e1->Copy(), 1);
    }

    static PTR_T Exp(const PTR_T &t1) {
        const VEC_E &ref = t1->getData();

        const PTR_T &temp = std::make_shared<Tensor>();
        for (int i = 0; i < ref.size(); i++) {
            temp->getData().push_back(Exp(ref[i]));
        }

        temp->setShape(t1->getShape());

        return temp;
    }
    static PTR_E Exp(const PTR_E &e1) {
        return std::make_shared<Element>(std::pow(EULER, e1->getData()), e1, std::pow(EULER, e1->getData()));
    }

    static PTR_T Log(const PTR_T &t1) {
        const VEC_E &ref = t1->getData();

        const PTR_T &temp = std::make_shared<Tensor>();
        for (int i = 0; i < ref.size(); i++) {
            temp->getData().push_back(Log(ref[i]));
        }

        temp->setShape(t1->getShape());

        return temp;
    }
    static PTR_E Log(const PTR_E &e1) {
        return std::make_shared<Element>(std::log(e1->getData()), e1, 1/e1->getData());
    }

    // %% ACTIVATIONS %%
    static PTR_T Softplus(const PTR_T &t1, const DTYPE beta=1, const DTYPE treshold=20) {
        const VEC_E &ref = t1->getData();

        const PTR_T &temp = std::make_shared<Tensor>();
        for (int i = 0; i < ref.size(); i++) {
            temp->getData().push_back(Softplus(ref[i], 1));
        }

        temp->setShape(t1->getShape());

        return temp;
    }
    static PTR_E Softplus(PTR_E e1, const DTYPE beta, const DTYPE treshold=20) {
        if (e1->getData() >= treshold) {
            return e1;
        }
        e1 = MulElement(Log(AddElement(Exp(MulElement(e1, beta)),1)),1/beta);

        return e1;
    }

    static PTR_T Softmax(const PTR_T &t1) {
        const VEC_E &ref = t1->getData();

        //: (A, B, C,...)
        // Copy : (B, N) -> exp(Ni)/sum(exp(N)) -> (B, N)

        PTR_T tensor_copy = t1->Copy();
        auto original_shape = tensor_copy->getShape();

        const int D = original_shape[original_shape.size() - 1];

        tensor_copy = ReShape(tensor_copy, {-1, D});

        VEC_E soft_vec;
        for (int b = 0; b < tensor_copy->getShape()[0]; ++b) {
            VEC_I start{b, 0};
            VEC_I end{b, D - 1};

            const PTR_T &sliced_t = TakeSlice(tensor_copy, start, end);
            const VEC_E &sliced = sliced_t->getData();

            VEC_E exps = {};
            for (int n = 0; n < D; n++) {
                exps.push_back(Exp(sliced[n]));
            }

            PTR_E sum = std::make_shared<Element>(.0);
            for (const PTR_E& e : exps) { sum = AddElement(sum, e); }

            for (PTR_E e : exps) {
                e = DivElement(e, sum);
                soft_vec.push_back(e);
            }
        }

        return std::make_shared<Tensor>(soft_vec, original_shape);
    }

    static PTR_T Sigmoid(const PTR_T &t1) {
        const VEC_E &ref = t1->getData();

        const PTR_T &temp = std::make_shared<Tensor>();
        for (int i = 0; i < ref.size(); i++) {
            temp->getData().push_back(Sigmoid(ref[i]));
        }

        temp->setShape(t1->getShape());

        return temp;
    }
    static PTR_E Sigmoid(const PTR_E &e1) {
        const PTR_E &tempexp = Exp(e1);
        const PTR_E &add = AddElement(tempexp, 1.0);

        return DivElement(tempexp, add);
    }

    static PTR_T ReLU(const PTR_T &t1) {
        const VEC_E &ref = t1->getData();

        const PTR_T &temp = std::make_shared<Tensor>();
        for (int i = 0; i < ref.size(); i++) {
            temp->getData().push_back(ReLU(ref[i]));
        }

        temp->setShape(t1->getShape());

        return temp;
    }
    static PTR_E ReLU(const PTR_E &e1) {
        if (e1->getData() > 0) {
            return Abs(e1);
        }
        return std::make_shared<Element>(.0,.0);
    }

    static PTR_T GeLU(const PTR_T &t1) {
        const VEC_E &ref = t1->getData();

        const PTR_T &temp = std::make_shared<Tensor>();
        for (int i = 0; i < ref.size(); i++) {
            temp->getData().push_back(GeLU(ref[i]));
        }

        temp->setShape(t1->getShape());

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

    static PTR_T SiLU(const PTR_T &t1) {
        const VEC_E &ref = t1->getData();

        const PTR_T &temp = std::make_shared<Tensor>();
        for (int i = 0; i < ref.size(); i++) {
            temp->getData().push_back(SiLU(ref[i]));
        }

        temp->setShape(t1->getShape());

        return temp;
    }
    static PTR_E SiLU(const PTR_E &e1) {
        const PTR_E &silued = Sigmoid(e1);

        return MulElement(silued, e1);
    }

    static PTR_T Tanh(const PTR_T &t1) {
        const VEC_E &ref = t1->getData();

        const PTR_T &temp = std::make_shared<Tensor>();
        for (int i = 0; i < ref.size(); i++) {
            temp->getData().push_back(Tanh(ref[i]));
        }

        temp->setShape(t1->getShape());

        return temp;
    }
    static PTR_E Tanh(const PTR_E &e1) {
        const PTR_E &nom = MulElement(SubElement(Exp(MulElement(e1, -2)), 1), -1);
        const PTR_E &denom = AddElement(Exp(MulElement(e1, -2)), 1);

        const PTR_E &result = DivElement(nom, denom);

        return result;
    }

    // %% LOSSES %%
    static PTR_T CategoricalCrossEntropy(const PTR_T &t1, const int index) {
        // probs : (B, N) -> N = Number of Classes
        const int class_index = index < 0 ? index + t1->getShape()[1] : index;

        VEC_E loss_vec;
        for (int i = 0; i < t1->getShape()[0]; i++) {
            VEC_I start{i, 0};
            VEC_I end{i, t1->getShape()[0]-1};

            PTR_E &sliced = TakeSlice(t1, start, end)->getData()[class_index]; // (N,)[Class_index] -> score
            sliced = MulElement(Log(AddElement(sliced, 0.00001)), -1);

            loss_vec.push_back(sliced);
        }

        return std::make_shared<Tensor>(loss_vec);
    }
    static PTR_E CategoricalCrossEntropy(const VEC_E &probs, const int index) {
        const int class_index = index < 0 ? index + probs.size() : index;

        return MulElement(Log(AddElement(probs[class_index], 0.00001)), -1);
    }

    static PTR_T BinaryCrossEntropy(const PTR_T &t1, const int index) {
        // t1 : (B, 2)
        const int class_index = index < 0 ? index + t1->getShape()[1] : index;

        return CategoricalCrossEntropy(t1, class_index);
    }
    static PTR_E BinaryCrossEntropy(const VEC_E &probs, const int index) {
        const int class_index = index < 0 ? index + probs.size() : index;

        return CategoricalCrossEntropy(probs, class_index);
    }

    static PTR_T MSE(const PTR_T &t1, const PTR_T &t2) {
        return Pow(FlexibleSub(t1, t2), 2);
    }
    static PTR_E MSE(const PTR_E &e1, const DTYPE y) {
        return Pow(SubElement(e1, y), 2);;
    }

    static PTR_T MAE(const PTR_T &t1, const PTR_T &t2){
        return Abs(FlexibleSub(t1, t2));
    }
    static PTR_E MAE(const PTR_E &e1, const DTYPE y) {
        return Abs(SubElement(e1, y));
    }

    // %% UILIY FUNCIONS %%
    static void Save(String path, const PTR_T &obj) {
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

        for (auto & i : obj->getData()) {
            data_str += std::to_string(i->getData());
            data_str += '?';
            data_str += std::to_string(i->getImData());
            data_str += '?';
            data_str += std::to_string(i->getGradient());
            data_str += '?';
            data_str += '|';
        }

        // cout << "%SAR_COMPLEX%   " << complex_str << "   %SAR_SHAPE%   " << shape_str << "   %SAR_WEIGHS%   " << weights_str <<"   %SAR_DAA%   " << data_str;

        file << complex_str << "\n" << shape_str << "\n" << weights_str << "\n" << std::to_string(obj->getData().size()) << "\n" << data_str << "\n";
    }

    static PTR_T Load(const String& path) {
        std::ifstream file(path);

        if (!file.is_open()) {
            throw std::runtime_error("File Issue!");
        }

        const String s((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        VEC_S elements = SplitByChar(s, '\n');

        bool complex = elements[0] == "1";
        VEC_I shape_vec;
        VEC_I weights_vec;
        VEC_E data_vec;

        VEC_S shape_str = SplitByChar(elements[1], '|');
        for (const auto & i : shape_str) {
            shape_vec.push_back(std::stoi(i));
        }

        VEC_S weights_str = SplitByChar(elements[2], '|');
        for (const auto & i : weights_str) {
            weights_vec.push_back(std::stoi(i));
        }

        VEC_S data_str = SplitByChar(elements[4], '|');
        for (const auto & i : data_str) {
            VEC_S element_str = SplitByChar(i, '?');

            DTYPE data = std::stold(element_str[0]);
            DTYPE im_data = std::stold(element_str[1]);
            DTYPE gradient = std::stold(element_str[2]);

            data_vec.push_back(std::make_shared<Element>(data, im_data, gradient));
        }

        return std::make_shared<Tensor>(data_vec, shape_vec, weights_vec, complex);
    }

    static PTR_T PadInput(PTR_T x, const int P_w, const int P_h) {
        const VEC_I &x_s = x->getShape();

        const int B = x_s[0];
        const int C = x_s[1];
        const int H = x_s[2];
        const int W = x_s[3];

        int leftSize = P_w / 2;
        int upSize = P_h / 2;

        int downSize = P_h - upSize;
        int rightSize = P_w - leftSize;

        // int _upSize = upSize * B * C * (W + rightSize);
        // int _leftSize = leftSize * B * C * (H + P_h);
        // int _downSize = downSize * B * C * W;
        // int _rightSize = rightSize * B * C * (H + downSize);

        auto down_pad = Minitorch::Zeros(downSize);
        down_pad = Minitorch::ReShape(down_pad, {1, 1, downSize, 1}); // {1, 1, DOWNSIZE, 1}
        auto repped_down_pad = Minitorch::Repeat(down_pad, {B, C, 1, 1});
        auto full_down_pad = Minitorch::Repeat(repped_down_pad, {1, 1, 1, W}); // {B, C, DOWNSIZE, W}

        auto right_pad = Minitorch::Zeros(rightSize);
        right_pad = Minitorch::ReShape(right_pad, {1, 1, 1, rightSize}); // {1, 1, 1, RIGHTSIZE}
        auto repped_right_pad = Minitorch::Repeat(right_pad, {B, C, 1, 1});
        auto full_right_pad = Minitorch::Repeat(repped_right_pad, {1, 1, (H + downSize), 1}); // {B, C, DOWNSIZE, W}

        PTR_T full_left_pad = nullptr;
        if (leftSize > 0) {
            auto left_pad = Minitorch::Zeros(leftSize);
            left_pad = Minitorch::ReShape(left_pad, {1, 1, 1, leftSize});
            auto repped_left_pad = Minitorch::Repeat(left_pad, {B, C, 1, 1});
            full_left_pad = Minitorch::Repeat(repped_left_pad, {1, 1, H + downSize, 1}); // {B, C, H, RIGHTSIZE}
        }

        PTR_T full_up_pad = nullptr;
        if (upSize > 0) {
            auto up_pad = Minitorch::Zeros(upSize);
            up_pad = Minitorch::ReShape(up_pad, {1, 1, upSize, 1});
            auto repped_up_pad = Minitorch::Repeat(up_pad, {B, C, 1, 1});
            full_up_pad = Minitorch::Repeat(repped_up_pad, {1, 1, 1, (W + P_w)}); // {B, C, UPSIZE, W}
        }

        auto downed = Minitorch::Concatenate(x, full_down_pad, 2);
        auto righted = Minitorch::Concatenate(downed, full_right_pad, 3);

        PTR_T lefted = nullptr;
        if (leftSize > 0) {
            lefted = Minitorch::Concatenate(full_left_pad, righted, 3);
        }
        else {
            lefted = righted;
        }

        PTR_T upped = nullptr;
        if (upSize > 0) {
            upped = Minitorch::Concatenate(full_up_pad, lefted, 2);
        }
        else {
            upped = lefted;
        }

        return upped;
    }

    static PTR_T Gather(const PTR_T &t1, const int dim, const PTR_T &indexes) {
        //A{3,5,5} - I{1,5,5}

        if (indexes->getShape().size() > t1->getShape().size()) {
            throw std::runtime_error("Bad Index!");
        }

        VEC_E out;
        for (int i = 0; i < indexes->getData().size(); ++i) {
            VEC_I index = CreateIndex(indexes, i);  // 0, 1, 2 - 0d
            index[dim] = TakeObject(indexes, i)->getData(); // -> 2, 0, 2

            out.push_back(TakeObject(t1, index));
        }

        return std::make_shared<Tensor>(out, indexes->getShape());
    }

    // 2 2 2 - 2 1 2 - d1
    // 2 2 2 - 2 1 2 - d1 -TRANSPOSE> 2 2 2 - 1 2 2 - d0 -> 3 2 2 -TRANSPOSE> 2 3 2
    static PTR_T Concatenate(const PTR_T &t1, const PTR_T &t2, int dim) {
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

        for (int i = 0; i < t1_shape.size(); ++i) {
            if (i != dim) {
                if (t1_shape[i] != t2_shape[i]) {
                    throw std::runtime_error("Incompatible Shapes!");
                }
            }
        }

        const PTR_T &ref_ = TakeTranspose(t1, dim, 0);
        const PTR_T &ref2_ = TakeTranspose(t2, dim, 0);

        VEC_I f_shape = ref_->getShape();
        f_shape[0] += ref2_->getShape()[0];

        const VEC_E &ref = ref_->getData();
        const VEC_E &ref2 = ref2_->getData();

        const int t1_size = ref.size();
        const int t2_size = ref2.size();

        VEC_E out;
        for (int i = 0; i < t1_size; i++) {
            out.push_back(ref[i]);
        }
        for (int i = 0; i < t2_size; i++) {
            out.push_back(ref2[i]);
        }

        PTR_T out_tensor = std::make_shared<Tensor>(out, f_shape);

        out_tensor = TakeTranspose(out_tensor, dim, 0);

        return out_tensor;
    }
    
    static PTR_T Where(const PTR_T &mask, DTYPE condition, DTYPE one, DTYPE zero) {
        VEC_E out;
        for (int i = 0; i < mask->Numel(); i++) {
            DTYPE temp_data = mask->getData()[i]->getData();

            if (temp_data == condition) {
                const PTR_E &temp = std::make_shared<Element>(one);
                out.push_back(temp);
            }
            else {
                const PTR_E &temp = std::make_shared<Element>(zero);
                out.push_back(temp);
            }
        }

        auto t_out = std::make_shared<Tensor>(out);
        t_out = Minitorch::ReShape(t_out, mask->getShape());

        return t_out;
    }

    static PTR_T Where(const PTR_T &mask, DTYPE one, DTYPE zero) {
        const VEC_E &ref = mask->getData();

        VEC_E out;
        for (int i = 0; i < mask->Numel(); i++) {
            const DTYPE temp_data = ref[i]->getData();

            if (temp_data > 0) {
                const PTR_E &temp = std::make_shared<Element>(one);
                out.push_back(temp);
            }
            else {
                const PTR_E &temp = std::make_shared<Element>(zero);
                out.push_back(temp);
            }
        }

        return std::make_shared<Tensor>(out, mask->getShape());
    }

    static PTR_T Narrow(const PTR_T &t1, int dim, int start, int length) {
        const int &size = t1->getShape().size();

        if (dim < 0) {
            dim = size + dim;
        }
        if (start < 0) {
            start = size + start;
        }
        if (dim < 0) {
            throw std::runtime_error("Bad Dimension!");
        }
        if (start < 0) {
            throw std::runtime_error("Bad Start!");
        }
        if (start + length >= t1->getShape()[dim]) {
            length = t1->getShape()[dim] - start - 1;
        }
        if (length < 1) {
            throw std::runtime_error("Bad Length!");
        }

        VEC_I start_indexes;
        VEC_I end_indexes;

        start_indexes.reserve(size);
        end_indexes.reserve(size);

        for (int i = 0; i < size; i++) {
            if (i != dim) {
                end_indexes.push_back(t1->getShape()[i] - 1); // (3,4,5) -all> (0,0,0) - (2,3,4)
                start_indexes.push_back(0);
            }
            else {
                end_indexes.push_back(length + start - 1);
                start_indexes.push_back(start);
            }
        }

        return std::make_shared<Tensor>(TakeSlice(t1, start_indexes, end_indexes));
    }

    // static void ElementWiseOpt(const PTR_T &t1, DTYPE(*op)(DTYPE));

    // 2, 9, 2 - c2 d1 -> <2, 4, <2, 1,
    static VEC_T Chunk(const PTR_T &t1, int dim, const int chunk_size)
    {
        if (dim < 0) {
            dim += t1->getShape().size();
        }
        if (dim < 0) {
            throw std::runtime_error("Bad Dimension!");
        }
        if (chunk_size >= t1->getShape()[dim]) {
            VEC_T temp{t1};

            return temp;
        }

        if (dim != 0) {
            const PTR_T transposed = TakeTranspose(t1, dim, 0);

            auto out = Chunk(transposed, 0, chunk_size);

            for(auto & i : out) {
                i = TakeTranspose(i, dim, 0);
            }

            return out;
        }

        const int chunk_dim_size = t1->getShape()[0];
        const int total_tensor_size = t1->getData().size();

        const VEC_E &ref = t1->getData();

        const bool perfect = chunk_dim_size % chunk_size == 0;
        const int n_iteration = 1 + chunk_dim_size / chunk_size;

        const int chunked_dim_size = chunk_dim_size / chunk_size;
        const int singular_number_elements = total_tensor_size / chunk_dim_size; // 2 1 2 toplam eleman sayisi;
        const int number_elements = singular_number_elements * chunked_dim_size; // -> singular ile toplam dimi carp normaller icin 2 1 2'den 4 tane var mesela;
        const int last_number_elements = singular_number_elements * (chunk_dim_size % chunk_size); // singularla sayiyi carp 2 1 2'den 1 tane kaldi;

        VEC_I full_shape = t1->getShape();
        full_shape[0] /= chunk_size;

        VEC_I last_shape = t1->getShape();
        last_shape[0] %= chunk_size;

        VEC_T out;
        for (int i = 0; i < n_iteration; ++i) {
            if (i != n_iteration - 1 || perfect) {
                VEC_E temp_data;

                for (int e = 0; e < number_elements; ++e) {
                    temp_data.push_back(ref[i * number_elements + e]->Copy());
                }

                out.push_back(std::make_shared<Tensor>(temp_data, full_shape));
            }
            else {
                VEC_E temp_data;

                for (int e = 0; e < last_number_elements; ++e) {
                    temp_data.push_back(ref[i * number_elements + e]->Copy());
                }

                out.push_back(std::make_shared<Tensor>(temp_data, last_shape));
            }
        }

        return out;
    }

    static PTR_T Vander(const PTR_T &t1, int n=-1) {
        const VEC_E &ref = t1->getData();
        const int size = ref.size();
        
        if (n < 1) { n = size; }

        VEC_E out{};
        for (int p = n; p > 0; --p) {
            for (int i = 0; i < size; ++i) {
                const PTR_E temp = std::make_shared<Element>(std::pow(ref[i]->getData(), p-1));
                out.push_back(temp);
            }
        }

        VEC_I shape{n, size};

        PTR_T temp = std::make_shared<Tensor>(out, shape);
        temp = TakeTranspose(temp, 0, 1);

        return temp;
    }

    static PTR_T LinSpace(const double start, const double end, const double length) {
        if (length < 1) {
            throw std::runtime_error("Bad Length!");
        }
        VEC_E out;

        const double hop_size = (end - start) / length;
        for (int i = 0; i < length; i++) {
            const PTR_E &temp = std::make_shared<Element>(start + i * hop_size);
            out.push_back(temp);
        }

        VEC_I shape;
        shape.reserve(1);

        shape.push_back(out.size());

        return std::make_shared<Tensor>(out, shape);
    }

    static PTR_T LogSpace(const int start, const int end) {
        VEC_E out;

        const double s = std::pow(2, start);
        const double hop_size = (std::pow(2, end) - s) / end - start;

        for (int i = 0; i < end - start; i++) {
            const PTR_E &temp = std::make_shared<Element>(s + i * hop_size);
            out.push_back(temp);
        }

        VEC_I shape{end - start};

        return std::make_shared<Tensor>(out, shape);
    }

    static PTR_T Zeros(const int n) {
        if (n < 1) {
            throw std::runtime_error("Bad Length!");
        }
        VEC_E out;

        for (int i = 0; i < n; i++) {
            const PTR_E &temp = std::make_shared<Element>(.0);
            out.push_back(temp);
        }

        PTR_T t = std::make_shared<Tensor>(out);

        return t;
    }

    static PTR_T Ones(const int n) {
        if (n < 1) {
            throw std::runtime_error("Bad Length!");
        }
        VEC_E out;

        for (int i = 0; i < n; i++) {
            const PTR_E &temp = std::make_shared<Element>(1.0);
            out.push_back(temp);
        }

        return std::make_shared<Tensor>(out);
    }

    static PTR_T Arange(const int start, const int end, const int T3) {
        VEC_E out;

        for (int i = start; i < end; i += T3) {
            out.push_back(std::make_shared<Element>(i));
        }

        VEC_I shape;
        shape.reserve(1);

        shape.push_back(out.size());

        return std::make_shared<Tensor>(out, shape);
    }

    static PTR_T Arange(const int start, const int end){
        if (end - start < 0) {
            throw std::runtime_error("Bad Length!");
        }
        VEC_E out;

        for (int i = start; i < end; ++i) {
            out.push_back(std::make_shared<Element>(i));
        }

        VEC_I shape;
        shape.reserve(1);

        shape.push_back(out.size());

        return std::make_shared<Tensor>(out, shape);
    }

    static PTR_T Diagonal(const int n) {
        if (n < 1) {
            throw std::runtime_error("Bad Length!");
        }
        VEC_E outer;

        for (int r = 0; r < n; r++) {
            for (int c = 0; c < n; c++) {
                if (r == c) {
                    const PTR_E &temp = std::make_shared<Element>(1.0);
                    outer.push_back(temp);
                }
                else {
                    const PTR_E &temp = std::make_shared<Element>(.0);
                    outer.push_back(temp);
                }
            }
        }

        VEC_I shape = {n, n};

        return std::make_shared<Tensor>(outer, shape);
    }

    static PTR_T Tril(const int n) {
        if (n < 1) {
            throw std::runtime_error("Bad Length!");
        }
        VEC_E out;

        for (int r = 1; r <= n; r++) {
            for (int c = 0; c < r; c++) {
                out.push_back(std::make_shared<Element>(1.0));
            }
            for (int c = 0; c < n - r; c++) {
                out.push_back(std::make_shared<Element>(.0));
            }
        }

        VEC_I shape{n, n};

        return std::make_shared<Tensor>(out, shape);
    }

    static PTR_T CreateOneHot(const int n, int index, const bool row=true) {
        if (n < 1) {
            throw std::runtime_error("Bad Length!");
        }
        if (index >= n) {
            throw std::runtime_error("Bad Index!");
        }
        if (index < 0) {
            index += n;
        }
        if (index < 0) {
            throw std::runtime_error("Bad Index!");
        }
        
        auto t1 = Zeros(n);

        VEC_I shape;
        shape.reserve(2);

        if (row) {
            shape = {1, n};
        }else {
            shape = {n, 1};
        }

        auto one = std::make_shared<Element>(1.0);

        t1 = ReShape(t1, shape);
        t1->SetObject({index}, one);

        return t1;
    }

    // %% INITIALIZATION %%
    static PTR_T HeUniformTensorInitialization(const int size, const int f_in, VEC_I shape = {}) {
        if (size < 1) {
            throw std::runtime_error("Bad Length!");
        }
        if (f_in < 1) {
            throw std::runtime_error("Bad F_IN!");
        }
        static std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<DTYPE> dist(-std::sqrt(6.0 / f_in), std::sqrt(6.0 / f_in));

        VEC_E elements;
        for (int i = 0; i < size; i++) {
            const DTYPE &u1 = dist(rng);

            elements.push_back(std::make_shared<Element>(u1));
        }

        if (shape.size() > 0) {
            return std::make_shared<Tensor>(elements, shape, false, true);
        }

        return std::make_shared<Tensor>(elements, false, true);
    }
};
