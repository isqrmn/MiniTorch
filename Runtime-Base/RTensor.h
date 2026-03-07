#pragma once

class Tensor {
    VEC_E data;

    VEC_I shape = {0};
    VEC_I index_weights = {1};

    bool complex = false;
    bool ParameterTensor = false;

public:
    PTR_T Copy() { return std::make_shared<Tensor>(*this); }

    explicit Tensor(const VEC_E &data, const VEC_I &shape,
                    const VEC_I &index_weights, const bool complex,
                    const bool ParameterTensor) : data(data), shape(shape), index_weights(index_weights),
                                                  complex(complex), ParameterTensor(ParameterTensor) {
    }

    explicit Tensor(const VEC_E &data, const VEC_I &shape,
                    const VEC_I &index_weights, const bool complex) : data(data), shape(shape),
        index_weights(index_weights), complex(complex) {
    }

    explicit Tensor(const VEC_E &data, const VEC_I &shape,
                    const VEC_I &index_weights) : data(data), shape(shape), index_weights(index_weights),
                                                             complex(false) {
    }

    explicit Tensor(const VEC_E &data, const VEC_I &shape,
                    const bool complex, const bool ParameterTensor) : data(data), shape(shape), complex(complex),
                                                                      ParameterTensor(ParameterTensor) {
        UpdateIndexWeights();
    }

    explicit Tensor(const VEC_E &data, VEC_I &&shape,
                    const bool complex, const bool ParameterTensor) : data(data), shape(shape), complex(complex),
                                                                      ParameterTensor(ParameterTensor) {
        UpdateIndexWeights();
    }

    explicit Tensor(const VEC_E &data, VEC_I &&shape,
                    const bool complex) : data(data), shape(shape), complex(complex) {
        UpdateIndexWeights();
    }

    explicit Tensor(const VEC_E &data, const VEC_I &shape,
                    const bool complex) : data(data), shape(shape), complex(complex) {
        UpdateIndexWeights();
    }

    explicit Tensor(const VEC_E &data, const bool complex,
                    const bool ParameterTensor) : data(data), complex(complex),
                                                  ParameterTensor(ParameterTensor), shape({data.size()}) { UpdateIndexWeights(); }

    explicit Tensor(const VEC_E &data, VEC_I &&shape) : data(data),
        shape(shape) { UpdateIndexWeights(); }

    explicit Tensor(const VEC_E &data, const VEC_I &shape) : data(data), shape(shape) { UpdateIndexWeights(); }

    explicit Tensor(const VEC_E &data, const bool complex) : data(data),
        complex(complex), shape({data.size()}) { UpdateIndexWeights(); }

    explicit Tensor(const VEC_E &data) : data(data), shape({data.size()}) { UpdateIndexWeights(); }

    explicit Tensor(const PTR_T &t1) : complex(t1->getComplex()),
                                                        ParameterTensor(t1->getParameterTensor()) {
        VEC_E elements;
        for (auto & i : t1->getData()) {
            elements.push_back(std::make_shared<Element>(i));
        }

        VEC_I shape;
        shape.reserve(t1->getShape().size());
        for (int i: t1->getShape()) { shape.push_back(i); }

        VEC_I index_weights;
        index_weights.reserve(t1->getIndexWeights().size());
        for (int i: t1->getIndexWeights()) { index_weights.push_back(i); }

        setShape(shape);
        setIndexWeights(index_weights);
        setData(elements);
    }

    explicit Tensor(const VEC_I &shape) : shape(shape) {
        this->UpdateIndexWeights();
    }

    explicit Tensor(): data({}), shape({}) {}

    explicit Tensor(const Tensor &t1) = default;

    explicit Tensor(Tensor &&t1) = default;

    static VEC_E ToElement(const VEC_D &elements) {
        VEC_E elements_;
        for (int d = 0; d < elements.size(); ++d) { elements_.push_back(std::make_shared<Element>(elements[d])); }

        return elements_;
    }

    static PTR_T Construct(const VEC_D &elements) {
        VEC_E elements_ = ToElement(elements);

        PTR_T ab1 = std::make_shared<Tensor>(elements_);

        return ab1;
    }

    static PTR_T Construct(VEC_D &&elements) {
        VEC_E elements_ = ToElement(elements);

        PTR_T ab1 = std::make_shared<Tensor>(elements_);

        return ab1;
    }

    static PTR_T Construct(const VEC_D &elements, const VEC_I &shape) {
        VEC_E elements_ = ToElement(elements);

        PTR_T ab1 = std::make_shared<Tensor>(elements_);
        ab1->setShape(shape);

        return ab1;
    }

    static PTR_T Construct(const VEC_D &elements, VEC_I &&shape) {
        VEC_E elements_ = ToElement(elements);

        PTR_T ab1 = std::make_shared<Tensor>(elements_);
        ab1->setShape(shape);

        return ab1;
    }

    static PTR_T Construct(VEC_D &&elements, const VEC_I &shape) {
        VEC_E elements_ = ToElement(elements);

        PTR_T ab1 = std::make_shared<Tensor>(elements_);
        ab1->setShape(shape);

        return ab1;
    }

    static PTR_T Construct(VEC_D &&elements, VEC_I &&shape) {
        VEC_E elements_ = ToElement(elements);

        PTR_T ab1 = std::make_shared<Tensor>(elements_);
        ab1->setShape(shape);

        return ab1;
    }

    static PTR_T Construct(const VEC_D &elements, const VEC_I &shape, const bool complex, const bool ParameterTensor) {
        VEC_E elements_ = ToElement(elements);

        PTR_T ab1 = std::make_shared<Tensor>(elements_);
        ab1->setShape(shape);
        ab1->setComplex(complex);
        ab1->setParameterTensor(ParameterTensor);

        return ab1;
    }

    static PTR_T Construct(const VEC_D &elements, VEC_I &&shape, const bool complex, const bool ParameterTensor) {
        VEC_E elements_ = ToElement(elements);

        PTR_T ab1 = std::make_shared<Tensor>(elements_);
        ab1->setShape(shape);
        ab1->setComplex(complex);
        ab1->setParameterTensor(ParameterTensor);

        return ab1;
    }

    static PTR_T Construct(VEC_D &&elements, const VEC_I &shape, const bool complex, const bool ParameterTensor) {
        VEC_E elements_ = ToElement(elements);

        PTR_T ab1 = std::make_shared<Tensor>(elements_);
        ab1->setShape(shape);
        ab1->setComplex(complex);
        ab1->setParameterTensor(ParameterTensor);

        return ab1;
    }

    static PTR_T Construct(VEC_D &&elements, VEC_I &&shape, const bool complex, const bool ParameterTensor) {
        VEC_E elements_ = ToElement(elements);

        PTR_T ab1 = std::make_shared<Tensor>(elements_);
        ab1->setShape(shape);
        ab1->setComplex(complex);
        ab1->setParameterTensor(ParameterTensor);

        return ab1;
    }

    VEC_E &getData() { return data; };

    VEC_I getIndexWeights() { return index_weights; }

    [[nodiscard]] bool getParameterTensor() const { return ParameterTensor; }

    VEC_I getShape() { return shape; }

    [[nodiscard]] bool getComplex() const { return complex; }

    void SetObject(const VEC_I &indexing, const PTR_E &e) {
        int index = 0;
        for (int i = 0; i < indexing.size(); i++) {
            index += indexing[i] * index_weights[i];
        }

        getData()[index] = e;
    }

    void SetObject(const int indexing, const PTR_E &e) { getData()[indexing] = e; }

    void setData(const VEC_E &data) { this->data = data; }

    void setIndexWeights(const VEC_I &weights) { index_weights = weights; }

    void setShape(const VEC_I &shape) {
        this->shape = shape;
        UpdateIndexWeights();
    }

    void setParameterTensor(const bool parameter) { ParameterTensor = parameter; }

    void setComplex(const bool complex) { this->complex = complex; }

    void UpdateIndexWeights() {
        VEC_I weights;
        weights.reserve(shape.size());

        int w = 1;
        for (int i = shape.size(); i > 0; --i) { // NOLINT(*-narrowing-conversions)
            weights.insert(weights.begin(), w);
            w *= shape[i - 1];
        }

        index_weights = weights;
    }

    PTR_E TakeObject(const VEC_I &indexing) {
        const VEC_E &ref = getData();
        const VEC_I &ref2 = getIndexWeights();

        int index = 0;
        for (int i=0; i<indexing.size(); i++) {
            index += indexing[i] * ref2[i];
        }

        return ref[index];
    }
    PTR_E TakeObject(const int indexing) {
        const VEC_E &ref = getData();

        return ref[indexing];
    }

    void PrintTensor() {
        const VEC_E &ref = getData();

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
                cout << "]";
            }

            for (int s = shape.size()-1; s > 0; --s) { // NOLINT(*-narrowing-conversions)
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

    int Numel() { return getData().size(); }

    bool All() {
        for (const auto &i : getData()) {
            if (i->getData() == 0.0) {
                return false;
            }
        }
        return true;
    }

    bool Any() {
        for (const auto & i : getData()) {
            if (i->getData() != 0.0) {
                return true;
            }
        }
        return false;
    }

    PTR_T operator==(const PTR_T &other) {
        VEC_E mask;
        for (int i = 0; i < other->getData().size(); i++) {
            const DTYPE inner_val = getData()[i]->getData();

            if (const DTYPE outer_val = getData()[i]->getData(); inner_val == outer_val) {
                PTR_E temp = std::make_shared<Element>(1.0);
                mask.push_back(temp);
            } else {
                PTR_E temp = std::make_shared<Element>(.0);
                mask.push_back(temp);
            }
        }

        return std::make_shared<Tensor>(mask, other->getShape());
    }

    PTR_T operator<=(const PTR_T &other) {
        VEC_E mask;
        for (int i = 0; i < other->getData().size(); i++) {
            const DTYPE inner_val = getData()[i]->getData();

            if (const DTYPE outer_val = getData()[i]->getData(); inner_val <= outer_val) {
                PTR_E temp = std::make_shared<Element>(1.0);
                mask.push_back(temp);
            } else {
                PTR_E temp = std::make_shared<Element>(.0);
                mask.push_back(temp);
            }
        }

        return std::make_shared<Tensor>(mask, other->getShape());
    }

    PTR_T operator>=(const PTR_T &other) {
        VEC_E mask;
        for (int i = 0; i < other->getData().size(); i++) {
            const DTYPE inner_val = getData()[i]->getData();

            if (const DTYPE outer_val = getData()[i]->getData(); inner_val >= outer_val) {
                PTR_E temp = std::make_shared<Element>(1.0);
                mask.push_back(temp);
            } else {
                PTR_E temp = std::make_shared<Element>(.0);
                mask.push_back(temp);
            }
        }

        return std::make_shared<Tensor>(mask, other->getShape());
    }

    PTR_T operator||(const PTR_T &other) {
        VEC_E mask;
        for (int i = 0; i < other->getData().size(); i++) {
            const DTYPE inner_val = getData()[i]->getData();

            if (const DTYPE outer_val = getData()[i]->getData(); inner_val != 0 || outer_val != 0) {
                PTR_E temp = std::make_shared<Element>(1.0);
                mask.push_back(temp);
            } else {
                PTR_E temp = std::make_shared<Element>(.0);
                mask.push_back(temp);
            }
        }

        return std::make_shared<Tensor>(mask, other->getShape());
    }

    PTR_T operator&&(const PTR_T &other) {
        VEC_E mask;
        for (int i = 0; i < other->getData().size(); i++) {
            const DTYPE inner_val = getData()[i]->getData();

            if (const DTYPE outer_val = getData()[i]->getData(); inner_val != 0 && outer_val != 0) {
                PTR_E temp = std::make_shared<Element>(1.0);
                mask.push_back(temp);
            } else {
                PTR_E temp = std::make_shared<Element>(.0);
                mask.push_back(temp);
            }
        }

        return std::make_shared<Tensor>(mask, other->getShape());
    }

    PTR_T operator<(const PTR_T &other) {
        VEC_E mask;
        for (int i = 0; i < other->getData().size(); i++) {
            const DTYPE inner_val = getData()[i]->getData();

            if (const DTYPE outer_val = getData()[i]->getData(); inner_val < outer_val) {
                PTR_E temp = std::make_shared<Element>(1.0);
                mask.push_back(temp);
            } else {
                PTR_E temp = std::make_shared<Element>(.0);
                mask.push_back(temp);
            }
        }

        return std::make_shared<Tensor>(mask, other->getShape());
    }

    PTR_T operator>(const PTR_T &other) {
        VEC_E mask;
        for (int i = 0; i < other->getData().size(); i++) {
            const DTYPE inner_val = getData()[i]->getData();

            if (const DTYPE outer_val = getData()[i]->getData(); inner_val > outer_val) {
                PTR_E temp = std::make_shared<Element>(1.0);
                mask.push_back(temp);
            } else {
                PTR_E temp = std::make_shared<Element>(.0);
                mask.push_back(temp);
            }
        }

        return std::make_shared<Tensor>(mask, other->getShape());
    }
};
