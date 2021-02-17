/**
 * @file point_distribution.cuh
 * @author Philip Klaus
 * @brief Contains code to distribute cloud points to their target leaf nodes
 */

#pragma once

#include "kernel_executor.cuh"
#include "kernel_structs.cuh"
#include "types.cuh"

namespace chunking {

/**
 * Distributes 3D points from the point cloud to the leaf nodes of the octree.
 * The CUDA kernel iteratively tests whether the target node is marked
 * as finished. If it is finished the point index is stored in the nodes
 * point-LUT else the kernel tests the next higher parent node.
 *
 * @tparam coordinateType
 * @param octree
 * @param dataLUT
 * @param denseToSparseLUT
 * @param tmpIndexRegister
 * @param cloud
 * @param gridding
 */
template <typename coordinateType>
__global__ void kernelDistributePoints (
        Chunk* octree,
        uint32_t* dataLUT,
        int* denseToSparseLUT,
        uint32_t* tmpIndexRegister,
        KernelStructs::Cloud cloud,
        KernelStructs::Gridding gridding)
{
    int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= cloud.points)
    {
        return;
    }

    Vector3<coordinateType>* point = reinterpret_cast<Vector3<coordinateType>*> (cloud.raw + index * cloud.dataStride);

    auto sparseVoxelIndex = denseToSparseLUT[mapPointToGrid<coordinateType> (point, gridding)];

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
                chunking::kernelDistributePoints<float>, config.threadAmount, std::forward<Arguments> (args)...);
    }
    else
    {
        return executeKernel (
                chunking::kernelDistributePoints<double>, config.threadAmount, std::forward<Arguments> (args)...);
    }
}
} // namespace Kernel
