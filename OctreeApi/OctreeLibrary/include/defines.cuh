//
// Created by KlausP on 16.10.2020.
//

#pragma once

#include <cstdint>

#include "cuda_runtime.h"
#include "spdlog/spdlog.h"



constexpr uint32_t BLOCK_SIZE_MAX = 1024;
constexpr uint32_t GRID_SIZE_MAX  = 65535;

constexpr uint32_t INVALID_INDEX = 4294967295; // Use the max of uint32_t as an invalid index value

#define gpuErrchk(ans)                                                                                                 \
    {                                                                                                                  \
        gpuAssert ((ans), __FILE__, __LINE__);                                                                         \
    }
inline void gpuAssert (cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        spdlog::error ("GPUassert: {:s} {:s} {:d}", cudaGetErrorString (code), file, line);
        if (abort)
            exit (code);
    }
}
