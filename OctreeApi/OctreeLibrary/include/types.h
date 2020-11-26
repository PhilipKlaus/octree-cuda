//
// Created by KlausP on 01.11.2020.
//

#ifndef POTREECONVERTERGPU_TYPES_H
#define POTREECONVERTERGPU_TYPES_H


#include <cstdint>

//https://stackoverflow.com/questions/19995440/c-cast-byte-array-to-struct
#pragma pack(push, 1)
struct Vector3
{
    float x, y, z;
};
#pragma pack(pop)

struct Vector3i
{
    uint32_t x, y, z;
};

struct Chunk {
    uint32_t pointCount;            // How many points does this chunk have
    uint32_t parentChunkIndex;      // Determines the INDEX of the parent CHUNK in the GRID - Only needed during Merging
    bool isFinished;                // Is this chunk finished (= not mergeable anymore)
    uint32_t chunkDataIndex;        // Determines the INDEX in the chunk data array -> for storing point data
    int childrenChunks[8];          // The INDICES of the children chunks in the GRID
    uint32_t childrenChunksCount;   // Denotes the amount of children chunks (which are not empty)
    bool isParent;                  // Denotes if Chunk is a parent or a leaf node
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
