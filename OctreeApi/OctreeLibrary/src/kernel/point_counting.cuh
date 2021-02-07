#pragma once

#include "octree_metadata.h"
#include "tools.cuh"
#include "types.cuh"
#include "kernel_executor.cuh"


namespace chunking {

template <typename coordinateType>
__global__ void kernelInitialPointCounting (
        uint8_t* cloud,
        uint32_t* densePointCount,
        int* denseToSparseLUT,
        uint32_t* sparseIndexCounter,
        PointCloudMetadata metadata,
        uint32_t gridSideLength)
{
    int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= metadata.pointAmount)
    {
        return;
    }

    Vector3<coordinateType>* point =
            reinterpret_cast<Vector3<coordinateType>*> (cloud + index * metadata.pointDataStride);

    // 1. Calculate the index within the dense grid
    auto denseVoxelIndex = tools::calculateGridIndex<coordinateType> (point, metadata, gridSideLength);

    // 2. Accumulate the counter within the dense cell
    auto oldIndex = atomicAdd ((densePointCount + denseVoxelIndex), 1);

    // 3. If the thread is the first one accumulating the counter within the cell
    // -> update the denseToSparseLUT
    if (oldIndex == 0)
    {
        auto sparseVoxelIndex             = atomicAdd (sparseIndexCounter, 1);
        denseToSparseLUT[denseVoxelIndex] = sparseVoxelIndex;
    }
}
} // namespace chunking


namespace Kernel {

template <typename... Arguments>
float initialPointCounting (KernelConfig config, Arguments&&... args)
{
    if (config.cloudType == CLOUD_FLOAT_UINT8_T)
    {
        return executeKernel (
                chunking::kernelInitialPointCounting<float>,
                config.threadAmount,
                std::forward<Arguments> (args)...);
    }
    else
    {
        return executeKernel (
                chunking::kernelInitialPointCounting<double>,
                config.threadAmount,
                std::forward<Arguments> (args)...);
    }
}
}
