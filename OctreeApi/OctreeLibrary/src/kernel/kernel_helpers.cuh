#pragma once

#include "kernel_executor.cuh"
#include "kernel_structs.cuh"
#include <cstdint>
#include <cuda_runtime_api.h>


// See OctreeConverter : chunker_countsort_laszip.cpp :131
template <typename coordinateType>
inline __device__ uint32_t
        mapPointToGrid (const Vector3<coordinateType>* point, const KernelStructs::Gridding& gridding)
{
    double t  = gridding.bbSize / gridding.gridSize;
    double uX = (point->x - gridding.bbMin.x) / t;
    double uY = (point->y - gridding.bbMin.y) / t;
    double uZ = (point->z - gridding.bbMin.z) / t;

    t           = gridding.gridSize - 1.0;
    uint64_t ix = static_cast<int64_t> (fmin (uX, t));
    uint64_t iy = static_cast<int64_t> (fmin (uY, t));
    uint64_t iz = static_cast<int64_t> (fmin (uZ, t));

    return static_cast<uint32_t> (ix + iy * gridding.gridSize + iz * gridding.gridSize * gridding.gridSize);
}


/**
 * Encodes a color vector in a single uint64.
 * @tparam colorType The datatype of the point colors.
 * @param color The color vector to be encoded.
 * @return The encoded color information.
 */
template <typename colorType>
inline __device__ uint64_t encodeColors (Vector3<colorType>* color)
{
    return (static_cast<uint64_t> (color->x) << 46) | (static_cast<uint64_t> (color->y) << 28) |
           static_cast<uint64_t> (color->z) << 10 | static_cast<uint64_t> (1);
}


/**
 * Encodes three color components (r,g,b) in a single uint64.
 * @param r The red color component.
 * @param g The green color component.
 * @param b The blue color component.
 * @return The encoded color information.
 */
inline __device__ uint64_t encodeColors (uint16_t r, uint16_t g, uint16_t b)
{
    return (static_cast<uint64_t> (r) << 46) | (static_cast<uint64_t> (g) << 28) | static_cast<uint64_t> (b) << 10 |
           static_cast<uint64_t> (1);
}

inline __device__ uint64_t
        calculateWritingPosition (Node* octree, uint32_t nodeIdx, int lastNode, const uint32_t* leafOffset)
{
    return (lastNode == -1) ? leafOffset[0] : octree[lastNode].dataIdx + octree[lastNode].pointCount;
}
