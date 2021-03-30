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
 * as finished. If it is finished the point is distributed, otherwise the kernel tests the next higher parent node.
 * During distribution, on the one side, the indices of the points are stored in a LUT.
 * On the other side, the point data (coordinates, colors) are directly written to the binary output buffer.
 *
 * @tparam coordinateType
 * @param octree
 * @param dataLUT
 * @param denseToSparseLUT
 * @param tmpIndexRegister
 * @param cloud
 * @param gridding
 */
template <typename coordinateType, typename colorType>
__global__ void kernelDistributePoints (
        Node* octree,
        PointLut* output,
        OutputBuffer* outputBuffer,
        const int* denseToSparseLUT,
        uint32_t* tmpIndexRegister,
        KernelStructs::Cloud cloud,
        KernelStructs::Gridding gridding)
{
    unsigned int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= cloud.points)
    {
        return;
    }

    // Fetch point coordinates and color
    uint8_t* srcPoint = cloud.raw + index * cloud.dataStride;
    auto* point       = reinterpret_cast<Vector3<coordinateType>*> (srcPoint);
    auto* color       = reinterpret_cast<Vector3<colorType>*> (srcPoint + 3 * sizeof (coordinateType));

    auto sparseVoxelIndex = denseToSparseLUT[mapPointToGrid<coordinateType> (point, gridding)];

    bool isFinished = octree[sparseVoxelIndex].isFinished;

    // Search for finished octree node to store point data
    while (!isFinished)
    {
        sparseVoxelIndex = octree[sparseVoxelIndex].parentNode;
        isFinished       = octree[sparseVoxelIndex].isFinished;
    }

    uint32_t dataIndexWithinChunk = atomicAdd (tmpIndexRegister + sparseVoxelIndex, 1);

    // Store point index
    *(output + octree[sparseVoxelIndex].dataIdx + dataIndexWithinChunk) = index;

    // Write coordinates and colors to output buffer
    OutputBuffer* out = outputBuffer + octree[sparseVoxelIndex].dataIdx + dataIndexWithinChunk;
    out->x            = static_cast<int32_t> (floor ((point->x - gridding.bbMin.x) * cloud.scaleFactor.x));
    out->y            = static_cast<int32_t> (floor ((point->y - gridding.bbMin.y) * cloud.scaleFactor.y));
    out->z            = static_cast<int32_t> (floor ((point->z - gridding.bbMin.z) * cloud.scaleFactor.z));
    out->r            = color->x;
    out->g            = color->y;
    out->b            = color->z;
}

} // namespace chunking

namespace Kernel {

template <typename... Arguments>
void distributePoints (KernelConfig config, Arguments&&... args)
{
    if (config.cloudType == CLOUD_FLOAT_UINT8_T)
    {
        return executeKernel (
                chunking::kernelDistributePoints<float, uint8_t>,
                config.threadAmount,
                config.name,
                std::forward<Arguments> (args)...);
    }
    else
    {
        return executeKernel (
                chunking::kernelDistributePoints<double, uint8_t>,
                config.threadAmount,
                config.name,
                std::forward<Arguments> (args)...);
    }
}
} // namespace Kernel
