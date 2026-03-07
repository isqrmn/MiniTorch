#pragma once

#include <memory>
#include <array>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <random>
#include <fstream>
#include <variant>

#define Minitorch CMiniTorch
#define Tensor CTensor
#define Element CElement

#define SPTR std::shared_ptr
#define PTR_E std::shared_ptr<Element>

#define ARR std::array
#define VEC_I std::vector<int>

#include "Element.h"
#include "CTensor.h"
#include "CMiniTorch.h"
