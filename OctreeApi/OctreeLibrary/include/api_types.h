//
// Created by KlausP on 20.01.2021.
//

#pragma once

#include <cstdint>


// ToDo: Check if pragma is necessary
#pragma pack(push, 1)
template <typename T>
struct Vector3
{
    T x, y, z;
};
#pragma pack(pop)


enum CloudType
{
    CLOUD_FLOAT_UINT8_T,
    CLOUD_DOUBLE_UINT8_T
};

template <typename coordinateType>
struct BoundingBox
{
    Vector3<coordinateType> min;
    Vector3<coordinateType> max;
};

template <typename coordinateType>
struct PointCloudMetadata
{
    uint32_t pointAmount;
    uint32_t pointDataStride;
    BoundingBox<coordinateType> bbCubic;
    BoundingBox<coordinateType> bbReal;
    Vector3<coordinateType> cloudOffset;
    Vector3<coordinateType> scale;
    CloudType cloudType;
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
