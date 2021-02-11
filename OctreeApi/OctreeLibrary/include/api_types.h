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

enum CloudMemory
{
    CLOUD_HOST,
    ClOUD_DEVICE
};

struct BoundingBox
{
    Vector3<double> min;
    Vector3<double> max;
};

struct PointCloudMetadata
{
    uint32_t pointAmount;
    uint32_t pointDataStride;
    BoundingBox bbCubic;
    Vector3<double> cloudOffset;
    Vector3<double> scale;
    CloudType cloudType;
    CloudMemory memoryType;

    double cubicSize () const
    {
        return bbCubic.max.x - bbCubic.min.x;
    }
};
