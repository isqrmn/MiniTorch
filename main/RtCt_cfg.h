#pragma once

#define USE_CTRT_CFG 1

#if USE_CTRT_CFG == 1
    #include "Global_define.h"

    #define RUNTIME_MODE 0
    #define COMPILE_MODE 1
    #define MIXED_MODE 2
    #define MODE RUNTIME_MODE

    #if MODE == RUNTIME_MODE
        #include "../Runtime-Base/R_naming_cfg.h"
    #endif

    #if MODE == COMPILE_MODE
        #include "../Compile-Base/C_naming_cfg.h"
    #endif

    #if MODE == MIXED_MODE

    #endif
#endif
