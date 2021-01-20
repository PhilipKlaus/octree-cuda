//
// Created by KlausP on 20.01.2021.
//

#pragma once

#include <cstdint>


#pragma pack(push, 1)
template <typename T>
struct Vector3
{
    T x, y, z;
};
#pragma pack(pop)

struct BoundingBox
{
    Vector3<double> minimum;
    Vector3<double> maximum;
};

struct PointCloudMetadata
{
    uint32_t pointAmount;
    uint32_t pointDataStride;
    BoundingBox boundingBox;
    Vector3<double> cloudOffset;
    Vector3<double> scale;
};

enum SubsamplingStrategy
{
    FIRST_POINT,
    RANDOM_POINT
};

enum GridSize
{
    GRID_512 = 512,
    GRID_256 = 256,
    GRID_128 = 128,
    GRID_64  = 64,
    GRID_32  = 32,
    GRID_16  = 16,
    GRID_8   = 8,
    GRID_4   = 4,
    GRID_2   = 2,
    GRID_1   = 1
};
