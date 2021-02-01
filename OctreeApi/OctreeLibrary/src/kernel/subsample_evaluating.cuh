#pragma once

#include "octree_metadata.h"
#include "tools.cuh"
#include "types.cuh"


namespace subsampling {

template <typename coordinateType>
__global__ void kernelEvaluateSubsamples (
        uint8_t* cloud,
        SubsampleConfig* subsampleData,
        uint32_t* densePointCount,
        int* denseToSparseLUT,
        uint32_t* sparseIndexCounter,
        PointCloudMetadata metadata,
        uint32_t gridSideLength,
        uint32_t accumulatedPoints)
{
    int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= accumulatedPoints)
    {
        return;
    }

    // Determine child index and pick appropriate LUT data
    uint32_t* childDataLUT     = nullptr;
    uint32_t childDataLUTStart = 0;

    for (int i = 0; i < 8; ++i)
    {
        if (index < subsampleData[i].pointOffsetUpper)
        {
            childDataLUT      = subsampleData[i].lutAdress;
            childDataLUTStart = subsampleData[i].lutStartIndex;
            index -= subsampleData[i].pointOffsetLower;
            break;
        }
    }

    Vector3<coordinateType>* point = reinterpret_cast<Vector3<coordinateType>*> (
            cloud + childDataLUT[childDataLUTStart + index] * metadata.pointDataStride);

    // 1. Calculate the index within the dense grid of the evaluateSubsamples
    auto denseVoxelIndex = tools::calculateGridIndex (point, metadata, gridSideLength);

    // 2. We are only interested in the first point within a cell
    auto oldIndex = atomicAdd ((densePointCount + denseVoxelIndex), 1);

    // 3. If the thread is the first one -> increase map from the dense grid to the sparse grid
    if (oldIndex == 0)
    {
        auto sparseVoxelIndex             = atomicAdd (sparseIndexCounter, 1);
        denseToSparseLUT[denseVoxelIndex] = sparseVoxelIndex;
    }
}
} // namespace subsampling
