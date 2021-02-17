#pragma once

#include "kernel_structs.cuh"
#include <cstdint>
#include <cuda_runtime_api.h>


// See OctreeConverter : chunker_countsort_laszip.cpp :131
template <typename coordinateType>
__device__ uint32_t mapPointToGrid (const Vector3<coordinateType>* point, const KernelStructs::Gridding& gridding)
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