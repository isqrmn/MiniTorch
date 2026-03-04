#pragma once

#include <memory>

class Element {
    DTYPE data;
    DTYPE gradient;
    DTYPE im_data;

    PTR_E back_path_0 = nullptr;
    DTYPE back_scalar_0;

    PTR_E back_path_1 = nullptr;
    DTYPE back_scalar_1;

public:
    [[nodiscard]] PTR_E Copy() const { return std::make_shared<Element>(std::make_shared<Element>(*this)); }

    explicit Element(const DTYPE data, const DTYPE im_data, const DTYPE gradient, const PTR_E &back_path_0, const DTYPE back_scalar_0, const PTR_E &back_path_1, const DTYPE back_scalar_1):
    data(data), gradient(gradient), im_data(im_data), back_path_0(back_path_0), back_scalar_0(back_scalar_0), back_path_1(back_path_1), back_scalar_1(back_scalar_1)
    {}

    explicit Element(const DTYPE data, const DTYPE im_data, const PTR_E &back_path_0, const DTYPE back_scalar_0, const PTR_E &back_path_1, const DTYPE back_scalar_1):
    data(data), gradient(.0), im_data(im_data), back_path_0(back_path_0), back_scalar_0(back_scalar_0), back_path_1(back_path_1), back_scalar_1(back_scalar_1)
    {}

    explicit Element(const DTYPE data, const PTR_E &back_path_0, const DTYPE back_scalar_0, const PTR_E &back_path_1, const DTYPE back_scalar_1):
    data(data), gradient(.0), im_data(.0), back_path_0(back_path_0), back_scalar_0(back_scalar_0), back_path_1(back_path_1), back_scalar_1(back_scalar_1)
    {}

    explicit Element(const DTYPE data, const DTYPE im_data, const DTYPE gradient, const PTR_E &back_path_0, const DTYPE back_scalar_0):
    data(data), gradient(gradient), im_data(im_data), back_path_0(back_path_0), back_scalar_0(back_scalar_0), back_scalar_1(0)
    {}

    explicit Element(const DTYPE data, const DTYPE im_data, const PTR_E &back_path_0, const DTYPE back_scalar_0):
    data(data), gradient(.0), im_data(im_data), back_path_0(back_path_0), back_scalar_0(back_scalar_0), back_scalar_1(0)
    {}

    explicit Element(const DTYPE data, const PTR_E &back_path_0, const DTYPE back_scalar_0):
    data(data), gradient(.0), im_data(.0), back_path_0(back_path_0), back_scalar_0(back_scalar_0), back_scalar_1(0)
    {}

    explicit Element(const DTYPE data, const DTYPE im_data, const DTYPE gradient):
    data(data), gradient(gradient), im_data(im_data), back_scalar_0(0), back_scalar_1(0)
    {}

    explicit Element(const DTYPE data, const DTYPE im_data):
    data(data), gradient(.0), im_data(im_data), back_scalar_0(0), back_scalar_1(0)
    {}

    explicit Element(const PTR_E &e):
    data(e->getData()), gradient(e->getGradient()), im_data(e->getImData()), back_path_0(e->getBackPath0()), back_scalar_0(e->getBackScalar0()), back_path_1(e->getBackPath1()), back_scalar_1(e->getBackScalar1())
    {}

    explicit Element(const DTYPE data):
    data(data), gradient(.0), im_data(.0), back_scalar_0(0), back_scalar_1(0)
    {}

    explicit Element() : data(.0), gradient(0), im_data(0), back_scalar_0(0), back_scalar_1(0) {}

    [[nodiscard]] PTR_E getBackPath0() const { return back_path_0; }

    [[nodiscard]] PTR_E getBackPath1() const { return back_path_1; }

    [[nodiscard]] DTYPE getBackScalar0() const { return this->back_scalar_0; }

    [[nodiscard]] DTYPE getBackScalar1() const { return this->back_scalar_1; }

    [[nodiscard]] DTYPE getGradient() const { return this->gradient; }

    [[nodiscard]] DTYPE getImData() const { return this->im_data; }

    [[nodiscard]] DTYPE getData() const { return this->data; }

    void setGradient(const DTYPE grad) { this->gradient = grad; }

    void incGradient(const DTYPE grad) { this->gradient += grad; }

    void setImData(const DTYPE data) { this->im_data = data; }

    void setData(const DTYPE data) { this->data = data; }

    bool operator<=(const PTR_E &other) const  {
        if (this->getData() <= other->getData()) {
            return true;
        }
        return false;
    }

    bool operator>=(const PTR_E &other) const  {
        if (this->getData() >= other->getData()) {
            return true;
        }
        return false;
    }

    bool operator<(const PTR_E &other) const  {
        if (this->getData() < other->getData()) {
            return true;
        }
        return false;
    }

    bool operator>(const PTR_E &other) const  {
        if (this->getData() > other->getData()) {
            return true;
        }
        return false;
    }

    // Element operator*(const Element &other) const;

    void apply_operation(DTYPE (*op)(DTYPE))  { this->data = op(this->data); }

    [[nodiscard]] PTR_E conj() const { return std::make_shared<Element>(this->data, this->im_data * -1); }

    void Autograd(const DTYPE dxdx=1.0)  {
        setGradient(getGradient() + dxdx);

        const DTYPE nowGrad = getGradient();

        if (this->back_path_0 != nullptr) {
            getBackPath0()->Autograd(nowGrad * back_scalar_0);
        }

        if (this->back_path_1 != nullptr) {
            getBackPath1()->Autograd(nowGrad * back_scalar_1);
        }
    }
};
