//
// Created by KlausP on 16.10.2020.
//

#ifndef OCTREECUDA_DEFINES_CUH
#define OCTREECUDA_DEFINES_CUH

#include <cstdint>

constexpr uint32_t BLOCK_SIZE_MAX = 1024;
constexpr uint32_t GRID_SIZE_MAX = 65535;

constexpr uint32_t INVALID_INDEX = 4294967295; // Use the maximum of uint32_t as an invalid index value

#endif //OCTREECUDA_DEFINES_CUH
