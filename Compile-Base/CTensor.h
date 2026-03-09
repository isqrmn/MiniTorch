#pragma once
#include <vector>

template<LLI T>
class Tensor {
    ARR<PTR_E, T> data = {};

    VEC_I shape = {};
    VEC_I index_weights = {};

    bool complex = false;
    bool ParameterTensor = false;

public:
    SPTR<Tensor<T>> Copy() { return std::make_shared<Tensor>(*this); }

    explicit Tensor(const ARR<PTR_E, T> &data, const VEC_I &shape={},
                    const VEC_I &index_weights={}, const bool complex=false,
                    const bool ParameterTensor=false) : data(data), shape(shape), index_weights(index_weights),
                                                  complex(complex), ParameterTensor(ParameterTensor) {
        if (shape.size() < 1) {
            setShape({T});
        }
        UpdateIndexWeights();
    }

    explicit Tensor(const ARR<PTR_E, T> &data, const VEC_I &shape,
                    const bool complex, const bool ParameterTensor) : data(data), shape(shape), complex(complex),
                                                                      ParameterTensor(ParameterTensor) {
        this->UpdateIndexWeights();
    }

    explicit Tensor(const ARR<PTR_E, T> &data, VEC_I &&shape,
                    const bool complex, const bool ParameterTensor) : data(data), shape(shape), complex(complex),
                                                                      ParameterTensor(ParameterTensor) {
        this->UpdateIndexWeights();
    }

    explicit Tensor(const ARR<PTR_E, T> &data, const VEC_I &shape,
                    const bool complex) : data(data), shape(shape), complex(complex) {
        this->UpdateIndexWeights();
    }

    explicit Tensor(const ARR<PTR_E, T> &data, const bool complex,
                    const bool ParameterTensor) : data(data), shape({T}), complex(complex),
                                                  ParameterTensor(ParameterTensor) { this->UpdateIndexWeights(); }

    explicit Tensor(const ARR<PTR_E, T> &data, const bool complex) : data(data), shape({T}),
        complex(complex) { this->UpdateIndexWeights(); }

    explicit Tensor(const SPTR<Tensor<T>> &t) : complex(t->getComplex()),
                                                        ParameterTensor(t->getParameterTensor()) {
        ARR<PTR_E, T> elements;
        for (int i = 0; i < t->getData().size(); ++i) {
            elements[i] = std::make_shared<Element>(t->getData()[i]);
        }

        VEC_I shape;
        shape.reserve(t->getShape().size());
        for (int i: t->getShape()) { shape.push_back(i); }

        VEC_I index_weights;
        index_weights.reserve(t->getIndexWeights().size());
        for (int i: t->getIndexWeights()) { index_weights.push_back(i); }

        this->setShape(shape);
        this->setIndexWeights(index_weights);
        this->setData(elements);
    }

    explicit Tensor(const Tensor &t1) = default;

    explicit Tensor(Tensor &&t1) = default;

    explicit Tensor() = default;

    static ARR<PTR_E, T> ToElement(const ARR<DTYPE, T> &elements) {
        ARR<PTR_E, T> elements_;
        for (int d = 0; d < elements.size(); ++d) { elements_[d] = std::make_shared<Element>(elements[d]); }

        return elements_;
    }

    static SPTR<Tensor<T>> Construct(const ARR<DTYPE, T> &elements) {
        ARR<PTR_E, T> elements_ = ToElement(elements);

        SPTR<Tensor<T>> ab1 = std::make_shared<Tensor>(elements_);

        return ab1;
    }

    static SPTR<Tensor<T>> Construct(ARR<DTYPE, T> &&elements) {
        ARR<PTR_E, T> elements_ = ToElement(elements);

        SPTR<Tensor<T>> ab1 = std::make_shared<Tensor>(elements_);

        return ab1;
    }

    static SPTR<Tensor<T>> Construct(const ARR<DTYPE, T> &elements, const VEC_I &shape) {
        ARR<PTR_E, T> elements_ = ToElement(elements);

        SPTR<Tensor<T>> ab1 = std::make_shared<Tensor<T>>(elements_);
        ab1->setShape(shape);

        return ab1;
    }

    static SPTR<Tensor<T>> Construct(const ARR<DTYPE, T> &elements, VEC_I &&shape) {
        ARR<PTR_E, T> elements_ = ToElement(elements);

        SPTR<Tensor<T>> ab1 = std::make_shared<Tensor>(elements_);
        ab1->setShape(shape);

        return ab1;
    }

    static SPTR<Tensor<T>> Construct(ARR<DTYPE, T> &&elements, const VEC_I &shape) {
        ARR<PTR_E, T> elements_ = ToElement(elements);

        SPTR<Tensor<T>> ab1 = std::make_shared<Tensor>(elements_);
        ab1->setShape(shape);

        return ab1;
    }

    static SPTR<Tensor<T>> Construct(ARR<DTYPE, T> &&elements, VEC_I &&shape) {
        ARR<PTR_E, T> elements_ = ToElement(elements);

        SPTR<Tensor<T>> ab1 = std::make_shared<Tensor>(elements_);
        ab1->setShape(shape);

        return ab1;
    }

    static SPTR<Tensor<T>> Construct(const ARR<DTYPE, T> &elements, const VEC_I &shape, const bool complex, const bool ParameterTensor) {
        ARR<PTR_E, T> elements_ = ToElement(elements);

        SPTR<Tensor<T>> ab1 = std::make_shared<Tensor>(elements_);
        ab1->setShape(shape);
        ab1->setComplex(complex);
        ab1->setParameterTensor(ParameterTensor);

        return ab1;
    }

    static SPTR<Tensor<T>> Construct(const ARR<DTYPE, T> &elements, VEC_I &&shape, const bool complex, const bool ParameterTensor) {
        ARR<PTR_E, T> elements_ = ToElement(elements);

        SPTR<Tensor<T>> ab1 = std::make_shared<Tensor>(elements_);
        ab1->setShape(shape);
        ab1->setComplex(complex);
        ab1->setParameterTensor(ParameterTensor);

        return ab1;
    }

    static SPTR<Tensor<T>> Construct(ARR<DTYPE, T> &&elements, const VEC_I &shape, const bool complex, const bool ParameterTensor) {
        ARR<PTR_E, T> elements_ = ToElement(elements);

        SPTR<Tensor<T>> ab1 = std::make_shared<Tensor>(elements_);
        ab1->setShape(shape);
        ab1->setComplex(complex);
        ab1->setParameterTensor(ParameterTensor);

        return ab1;
    }

    static SPTR<Tensor<T>> Construct(ARR<DTYPE, T> &&elements, VEC_I &&shape, const bool complex, const bool ParameterTensor) {
        ARR<PTR_E, T> elements_ = ToElement(elements);

        SPTR<Tensor<T>> ab1 = std::make_shared<Tensor>(elements_);
        ab1->setShape(shape);
        ab1->setComplex(complex);
        ab1->setParameterTensor(ParameterTensor);

        return ab1;
    }

    ARR<PTR_E, T> &getData() { return this->data; };

    VEC_I getIndexWeights() { return this->index_weights; }

    bool getParameterTensor() const { return this->ParameterTensor; }

    VEC_I getShape() { return this->shape; }

    bool getComplex() const { return this->complex; }

    void SetObject(const VEC_I &indexing, const PTR_E &e) {
        int index = 0;
        for (int i = 0; i < indexing.size(); i++) {
            index += indexing[i] * this->index_weights[i];
        }

        this->getData()[index] = e;
    }

    void SetObject(int indexing, const PTR_E &e) { this->getData()[indexing] = e; }

    void setData(const ARR<PTR_E, T> &data) { this->data = data; }

    void setIndexWeights(const VEC_I &weights) { this->index_weights = weights; }

    void setShape(const VEC_I &shape) {
        this->shape = shape;
        this->UpdateIndexWeights();
    }

    void setParameterTensor(bool parameter) { this->ParameterTensor = parameter; }

    void setComplex(const bool complex) { this->complex = complex; }

    void UpdateIndexWeights() {
        VEC_I weights;
        weights.reserve(shape.size());

        int w = 1;
        for (int i = this->shape.size(); i > 0; --i) {
            weights.insert(weights.begin(), w);
            w *= this->shape[i - 1];
        }

        this->index_weights = weights;
    }

    PTR_E TakeObject(const VEC_I &indexing) {
        const ARR<PTR_E, T> &ref = getData();
        const VEC_I &ref2 = getIndexWeights();

        int index = 0;
        for (int i=0; i<indexing.size(); i++) {
            index += indexing[i] * ref2[i];
        }

        return ref[index];
    }
    PTR_E TakeObject(int indexing) {
        const ARR<PTR_E, T> &ref = getData();

        return ref[indexing];
    }

    void PrintTensor() {
        const ARR<PTR_E, T> &ref = getData();

        // (B, 1, C, D)

        const VEC_I &shape = getShape();

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

            if ((i+1) % (shape[shape.size()-1]) == 0) { cout << TakeObject(i)->getData(); }
            else { cout << TakeObject(i)->getData() << " "; }

            if ((i+1) % (shape[shape.size()-1]) == 0) {
                if (i != mymax-1) { cout << "]"; }
                else { cout << "]"; }
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

    int Numel() { return this->getData().size(); }

    bool All() {
        for (int i = 0; i < this->getData().size(); i++) {
            if ((this->getData())[i]->getData() == 0.0) {
                return false;
            }
        }
        return true;
    }

    bool Any() {
        for (int i = 0; i < this->getData().size(); i++) {
            if ((this->getData()[i])->getData() != 0.0) {
                return true;
            }
        }
        return false;
    }

    SPTR<Tensor<T>> operator==(const SPTR<Tensor<T>> &other) {
        ARR<PTR_E, T> mask;
        for (int i = 0; i < other->getData().size(); i++) {
            const DTYPE inner_val = this->getData()[i]->getData();

            if (const DTYPE outer_val = this->getData()[i]->getData(); inner_val == outer_val) {
                PTR_E temp = std::make_shared<Element>(1.0);
                mask[i] = temp;
            } else {
                PTR_E temp = std::make_shared<Element>(.0);
                mask[i] = temp;
            }
        }

        return std::make_shared<Tensor>(mask, other->getShape());
    }

    SPTR<Tensor<T>> operator<=(const SPTR<Tensor<T>> &other) {
        ARR<PTR_E, T> mask;
        for (int i = 0; i < other->getData().size(); i++) {
            const DTYPE inner_val = this->getData()[i]->getData();

            if (const DTYPE outer_val = this->getData()[i]->getData(); inner_val <= outer_val) {
                PTR_E temp = std::make_shared<Element>(1.0);
                mask[i] = temp;
            } else {
                PTR_E temp = std::make_shared<Element>(.0);
                mask[i] = temp;
            }
        }

        return std::make_shared<Tensor>(mask, other->getShape());
    }

    SPTR<Tensor<T>> operator>=(const SPTR<Tensor<T>> &other) {
        ARR<PTR_E, T> mask;
        for (int i = 0; i < other->getData().size(); i++) {
            const DTYPE inner_val = this->getData()[i]->getData();

            if (const DTYPE outer_val = this->getData()[i]->getData(); inner_val >= outer_val) {
                PTR_E temp = std::make_shared<Element>(1.0);
                mask[i] = temp;
            } else {
                PTR_E temp = std::make_shared<Element>(.0);
                mask[i] = temp;
            }
        }

        return std::make_shared<Tensor>(mask, other->getShape());
    }

    SPTR<Tensor<T>> operator||(const SPTR<Tensor<T>> &other) {
        ARR<PTR_E, T> mask;
        for (int i = 0; i < other->getData().size(); i++) {
            const DTYPE inner_val = this->getData()[i]->getData();

            if (const DTYPE outer_val = this->getData()[i]->getData(); inner_val != 0 || outer_val != 0) {
                PTR_E temp = std::make_shared<Element>(1.0);
                mask[i] = temp;
            } else {
                PTR_E temp = std::make_shared<Element>(.0);
                mask[i] = temp;
            }
        }

        return std::make_shared<Tensor>(mask, other->getShape());
    }

    SPTR<Tensor<T>> operator&&(const SPTR<Tensor<T>> &other) {
        ARR<PTR_E, T> mask;
        for (int i = 0; i < other->getData().size(); i++) {
            const DTYPE inner_val = this->getData()[i]->getData();

            if (const DTYPE outer_val = this->getData()[i]->getData(); inner_val != 0 && outer_val != 0) {
                PTR_E temp = std::make_shared<Element>(1.0);
                mask[i] = temp;
            } else {
                PTR_E temp = std::make_shared<Element>(.0);
                mask[i] = temp;
            }
        }

        return std::make_shared<Tensor>(mask, other->getShape());
    }

    SPTR<Tensor<T>> operator<(const SPTR<Tensor<T>> &other) {
        ARR<PTR_E, T> mask;
        for (int i = 0; i < other->getData().size(); i++) {
            const DTYPE inner_val = this->getData()[i]->getData();

            if (const DTYPE outer_val = this->getData()[i]->getData(); inner_val < outer_val) {
                PTR_E temp = std::make_shared<Element>(1.0);
                mask[i] = temp;
            } else {
                PTR_E temp = std::make_shared<Element>(.0);
                mask[i] = temp;
            }
        }

        return std::make_shared<Tensor>(mask, other->getShape());
    }

    SPTR<Tensor<T>> operator>(const SPTR<Tensor<T>> &other) {
        ARR<PTR_E, T> mask;
        for (int i = 0; i < other->getData().size(); i++) {
            const DTYPE inner_val = this->getData()[i]->getData();

            if (const DTYPE outer_val = this->getData()[i]->getData(); inner_val > outer_val) {
                PTR_E temp = std::make_shared<Element>(1.0);
                mask[i] = temp;
            } else {
                PTR_E temp = std::make_shared<Element>(.0);
                mask[i] = temp;
            }
        }

        return std::make_shared<Tensor>(mask, other->getShape());
    }
};
