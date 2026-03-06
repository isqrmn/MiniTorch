#pragma once

#define USE_CTRT_CFG 1

#if USE_CTRT_CFG == 1

#define RUNTIME_MODE 0
#define COMPILE_MODE 1
#define MODE RUNTIME_MODE

#include "Global_define.h"

#if MODE == RUNTIME_MODE

#include "../Runtime-Base/R_naming_cfg.h"
#include "../Runtime-Base/RMiniTorch.h"

#define Minitorch RMiniTorch
#define Tensor RTensor

#endif

#if MODE == COMPILE_MODE

#include "../Compile-Base/C_naming_cfg.h"
#include "../Compile-Base/CMiniTorch.h"

#define Minitorch CMiniTorch
#define Tensor CTensor

#endif

#if MODE == RUNTIME_MODE

#define Tensor RTensor
#include "../Runtime-Base/R_naming_cfg.h"

#define Tensor CTensor
#include "../Compile-Base/C_naming_cfg.h"

#include "../Runtime-Base/RMiniTorch.h"
#include "../Compile-Base/CMiniTorch.h"

#endif

#endif
