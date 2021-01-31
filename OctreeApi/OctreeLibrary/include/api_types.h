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
    Vector3<float> scale;
    CloudType cloudType;
};

enum SubsamplingStrategy
{
    FIRST_POINT,
    RANDOM_POINT
};
