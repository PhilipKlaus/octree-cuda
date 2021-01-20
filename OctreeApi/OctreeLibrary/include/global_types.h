//
// Created by KlausP on 01.11.2020.
//

#pragma once

#include <cstdint>
#include <api_types.h>


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
    PointCloudMetadata cloudMetadata; // The cloud metadata;
    SubsamplingStrategy strategy;     // The subsampling strategy
};


#pragma pack(push, 1)
struct Chunk
{
    uint32_t pointCount;       // How many points does this chunk have
    uint32_t parentChunkIndex; // Determines the INDEX of the parent CHUNK in the GRID - Only needed during Merging
    bool isFinished;           // Is this chunk finished (= not mergeable anymore)
    uint32_t chunkDataIndex;   // Determines the INDEX in the chunk data array -> for storing point data
    int childrenChunks[8];     // The INDICES of the children chunks in the GRID
    bool isParent;             // Denotes if Chunk is a parent or a leaf node
};
#pragma pack(pop)
