#pragma once

class OptimizerClass {
public:
    static std::vector<std::shared_ptr<OptimizerClass>> opts;

    virtual ~OptimizerClass() = default;

    virtual DTYPE forward(PTR_E x) = 0;

    virtual DTYPE operator()(PTR_E x) { return forward(x); }
    
    virtual void step(const PTR_E &x, int i=0) = 0;
};
