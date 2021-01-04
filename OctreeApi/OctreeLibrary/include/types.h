//
// Created by KlausP on 01.11.2020.
//

#pragma once

#include <cstdint>

//https://stackoverflow.com/questions/19995440/c-cast-byte-array-to-struct
// https://stackoverflow.com/questions/2448242/struct-with-template-variables-in-c
#pragma pack(push, 1)
template <typename T>
struct CoordinateVector {
    T x, y, z;
};
#pragma pack(pop)


// ToDo: Maybe full double support?
struct BoundingBox {
    CoordinateVector<float> minimum;
    CoordinateVector<float> maximum;
};


// ToDo: Maybe full double support?
struct PointCloudMetadata {
    uint32_t pointAmount;
    uint32_t pointDataStride;
    BoundingBox boundingBox;
    CoordinateVector<float> cloudOffset;
    CoordinateVector<float> scale;
};


struct OctreeMetadata {

    uint32_t depth;                     // The depth of the octree // ToDo: -1
    uint32_t chunkingGrid;              // Side length of the grid used for chunking
    uint32_t subsamplingGrid;           // Side length of the grid used for subsampling
    uint32_t nodeAmountSparse;          // The actual amount of sparse nodes (amount leafs + amount parents)
    uint32_t leafNodeAmount;            // The amount of child nodes
    uint32_t parentNodeAmount;          // The amount of parent nodes
    uint32_t nodeAmountDense;           // The theoretical amount of dense nodes
    uint32_t mergingThreshold;          // Threshold specifying the (theoretical) minimum sum of points in 8 adjacent cells
    float meanPointsPerLeafNode;        // Mean points per leaf node
    float stdevPointsPerLeafNode;       // Standard deviation of points per leaf node
    uint32_t minPointsPerNode;
    uint32_t maxPointsPerNode;
    PointCloudMetadata cloudMetadata;   // The cloud metadata;
};

enum GridSize {
    GRID_512 = 512,
    GRID_256 = 256,
    GRID_128 = 128,
    GRID_64 = 64,
    GRID_32 = 32,
    GRID_16 = 16,
    GRID_8 = 8,
    GRID_4 = 4,
    GRID_2 = 2,
    GRID_1 = 1
};


#pragma pack(push, 1)
struct Chunk {
    uint32_t pointCount;            // How many points does this chunk have
    uint32_t parentChunkIndex;      // Determines the INDEX of the parent CHUNK in the GRID - Only needed during Merging
    bool isFinished;                // Is this chunk finished (= not mergeable anymore)
    uint32_t chunkDataIndex;        // Determines the INDEX in the chunk data array -> for storing point data
    int childrenChunks[8];          // The INDICES of the children chunks in the GRID
    bool isParent;                  // Denotes if Chunk is a parent or a leaf node
};
#pragma pack(pop)

