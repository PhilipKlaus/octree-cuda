//
// Created by KlausP on 16.10.2020.
//

#ifndef OCTREECUDA_DEFINES_CUH
#define OCTREECUDA_DEFINES_CUH

#include <cstdint>

constexpr uint64_t BLOCK_SIZE_MAX = 1024;
constexpr uint64_t GRID_SIZE_MAX = 65535;

constexpr uint64_t INVALID_INDEX = 18446744073709551615; // Use the maximum of uint64_t as an invalid index value

#endif //OCTREECUDA_DEFINES_CUH
