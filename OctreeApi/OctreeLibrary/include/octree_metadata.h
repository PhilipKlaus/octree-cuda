//
// Created by KlausP on 01.11.2020.
//

#pragma once

#include <cstdint>
#include <api_types.h>


template <typename coordinateType>
struct OctreeMetadata
{
    uint32_t depth;               // The depth of the octree // ToDo: -1
    uint32_t chunkingGrid;        // Side length of the grid used for chunking
    uint32_t subsamplingGrid;     // Side length of the grid used for subsampling
    uint32_t nodeAmountSparse;    // The actual amount of sparse nodes (amount leafs + amount parents)
    uint32_t leafNodeAmount;      // The amount of child nodes
    uint32_t parentNodeAmount;    // The amount of parent nodes
    uint32_t nodeAmountDense;     // The theoretical amount of dense nodes
    uint32_t mergingThreshold;    // Threshold specifying the (theoretical) minimum sum of points in 8 adjacent cells
    uint32_t absorbedNodes;       // Nodes completely absorbed during subsampling
    float meanPointsPerLeafNode;  // Mean points per leaf node
    float stdevPointsPerLeafNode; // Standard deviation of points per leaf node
    uint32_t minPointsPerNode;    // Minimum amount of points in a node
    uint32_t maxPointsPerNode;    // Maximum amount of points in a node
    PointCloudMetadata<coordinateType> cloudMetadata; // The cloud metadata;
    SubsamplingStrategy strategy;     // The subsampling strategy
};
