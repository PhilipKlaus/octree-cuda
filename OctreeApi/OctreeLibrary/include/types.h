//
// Created by KlausP on 01.11.2020.
//

#ifndef POTREECONVERTERGPU_TYPES_H
#define POTREECONVERTERGPU_TYPES_H


#include <cstdint>
#include <map>

namespace OctreeTypes {

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


}


//https://stackoverflow.com/questions/19995440/c-cast-byte-array-to-struct
#pragma pack(push, 1)
struct Vector3
{
    float x, y, z;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct Vector3i
{
    uint32_t x, y, z;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct Chunk {
    uint32_t pointCount;            // How many points does this chunk have
    uint32_t parentChunkIndex;      // Determines the INDEX of the parent CHUNK in the GRID - Only needed during Merging
    bool isFinished;                // Is this chunk finished (= not mergeable anymore)
    uint32_t chunkDataIndex;        // Determines the INDEX in the chunk data array -> for storing point data
    int childrenChunks[8];          // The INDICES of the children chunks in the GRID
    uint32_t childrenChunksCount;   // Denotes the amount of children chunks (which are not empty)
    bool isParent;                  // Denotes if Chunk is a parent or a leaf node
};
#pragma pack(pop)

struct BoundingBox {
    Vector3 minimum;
    Vector3 maximum;
};

struct PointCloudMetadata {
    uint32_t pointAmount;
    uint32_t pointDataStride;
    BoundingBox boundingBox;
    Vector3 cloudOffset;
    Vector3 scale;
};

// https://stackoverflow.com/questions/2448242/struct-with-template-variables-in-c
#pragma pack(push, 1)
template <typename T>
struct CoordinateVector {
    T x, y, z;
};
#pragma pack(pop)

#pragma pack(push, 1)
template <typename T>
struct ColorVector {
    T r, g, b;
};
#pragma pack(pop)

#endif //POTREECONVERTERGPU_TYPES_H
