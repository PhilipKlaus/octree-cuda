//
// Created by KlausP on 01.11.2020.
//

#ifndef POTREECONVERTERGPU_TYPES_H
#define POTREECONVERTERGPU_TYPES_H


#include <cstdint>

struct Vector3
{
    float x, y, z;
};

struct Vector3i
{
    uint32_t x, y, z;
};

struct Chunk {
    uint32_t pointCount;        // How many points does this chunk have
    uint32_t parentChunkIndex;  // Determines the INDEX of the parent CHUNK in the GRID - Only needed during Merging
    bool isFinished;            // Is this chunk finished (= not mergeable anymore)
    uint32_t chunkDataIndex;    // Determines the INDEX in the chunk data array -> for storing point data
    int childrenChunks[8];      // The INDICES of the children chunks in the GRID
    uint32_t childrenChunksCount;
};

struct BoundingBox {
    Vector3 minimum;
    Vector3 maximum;
};

struct PointCloudMetadata {
    uint32_t pointAmount;
    BoundingBox boundingBox;
    Vector3 cloudOffset;
    Vector3 scale;
};

#endif //POTREECONVERTERGPU_TYPES_H
