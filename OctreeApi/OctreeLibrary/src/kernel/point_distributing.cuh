#pragma once

#include "octree_metadata.h"
#include "tools.cuh"
#include "types.cuh"
#include "kernel_executor.cuh"


namespace chunking {

template <typename coordinateType>
__global__ void kernelDistributePoints (
        Chunk* octree,
        uint8_t* cloud,
        uint32_t* dataLUT,
        int* denseToSparseLUT,
        uint32_t* tmpIndexRegister,
        PointCloudMetadata metadata,
        uint32_t gridSize)
{
    int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= metadata.pointAmount)
    {
        return;
    }

    Vector3<coordinateType>* point =
            reinterpret_cast<Vector3<coordinateType>*> (cloud + index * metadata.pointDataStride);

    auto denseVoxelIndex  = tools::calculateGridIndex (point, metadata, gridSize);
    auto sparseVoxelIndex = denseToSparseLUT[denseVoxelIndex];

    bool isFinished = octree[sparseVoxelIndex].isFinished;

    while (!isFinished)
    {
        sparseVoxelIndex = octree[sparseVoxelIndex].parentChunkIndex;
        isFinished       = octree[sparseVoxelIndex].isFinished;
    }

    uint32_t dataIndexWithinChunk = atomicAdd (tmpIndexRegister + sparseVoxelIndex, 1);
    dataLUT[octree[sparseVoxelIndex].chunkDataIndex + dataIndexWithinChunk] = index;
}

} // namespace chunking

namespace Kernel {

template <typename... Arguments>
float distributePoints (KernelConfig config, Arguments&&... args)
{
    if (config.cloudType == CLOUD_FLOAT_UINT8_T)
    {
        return executeKernel (
                chunking::kernelDistributePoints<float>,
                config.threadAmount,
                std::forward<Arguments> (args)...);
    }
    else
    {
        return executeKernel (
                chunking::kernelDistributePoints<double>,
                config.threadAmount,
                std::forward<Arguments> (args)...);
    }
}
}
