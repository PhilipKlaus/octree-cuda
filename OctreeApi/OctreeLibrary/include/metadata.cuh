//
// Created by KlausP on 01.11.2020.
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

struct SubsampleMetadata
{
    uint32_t subsamplingGrid;
    bool performAveraging;
    bool useReplacementScheme;
};

struct NodeStatistics
{
    uint32_t leafNodeAmount;      // The amount of child nodes
    uint32_t parentNodeAmount;    // The amount of parent nodes
    float meanPointsPerLeafNode;  // Mean points per leaf node
    float stdevPointsPerLeafNode; // Standard deviation of points per leaf node
    uint32_t minPointsPerNode;    // Minimum amount of points in a node
    uint32_t maxPointsPerNode;    // Maximum amount of points in a node
};

struct OctreeMetadata
{
    uint32_t depth;            // The depth of the octree
    uint32_t chunkingGrid;     // Side length of the grid used for chunking
    uint32_t nodeAmountSparse; // The actual amount of sparse nodes (amount leafs + amount parents)
    uint32_t nodeAmountDense;  // The theoretical amount of dense nodes
    uint32_t mergingThreshold; // Threshold specifying the (theoretical) min sum of points in 8 adjacent cells
};
