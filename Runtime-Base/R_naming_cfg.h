#pragma once

#include <memory>
#include <cmath>
#include <random>
#include <fstream>
#include <vector>
#include <stdexcept>

#define String std::string

#define Minitorch RMiniTorch
#define Tensor RTensor
#define Element RElement

#define PTR_T std::shared_ptr<Tensor>
#define PTR_E std::shared_ptr<Element>

#define VEC_I std::vector<int>
#define VEC_S std::vector<String>
#define VEC_E std::vector<PTR_E>
#define VEC_T std::vector<PTR_T>
#define VEC_D std::vector<DTYPE>

#include "Element.h"
#include "RTensor.h"
#include "RMiniTorch.h"
