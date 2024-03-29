/**
 * @file point_counting.cuh
 * @author Philip Klaus
 * @brief Contains code to perform the initial point counting
 */

#pragma once

#include "kernel_executor.cuh"
#include "kernel_helpers.cuh"
#include "kernel_structs.cuh"
#include "types.cuh"

namespace chunking {

/**
 * Counts the amount of points per cell in a 3D counting grid.
 * Furthermore, the amount of occupied (non-empty) cells within
 * the grid is evaluated. For each occupied cell an entry in a
 * dense-to-sparse LUT is created.
 *
 * @tparam coordinateType The data type of the 3D coordinates in the cloud.
 * @param countingGrid Holds the amount of points per cell (dense).
 * @param filledNodeCounter Holds the amount of filled (non-empty) cells (sparse).
 * @param denseToSparseLUT Holds the dense-to-sparse node mapping.
 * @param cloud Holds point cloud related data.
 * @param gridding Holds gridding related data.
 */
template <typename coordinateType>
__global__ void kernelPointCounting (
        uint32_t* countingGrid,
        uint32_t* filledNodeCounter,
        int* denseToSparseLUT,
        KernelStructs::Cloud cloud,
        KernelStructs::Gridding gridding)
{
    unsigned int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= cloud.points)
    {
        return;
    }

    auto* point = reinterpret_cast<Vector3<coordinateType>*> (cloud.raw + index * cloud.dataStride);

    auto denseIndex = mapPointToGrid<coordinateType> (point, gridding);
    auto previous   = atomicAdd ((countingGrid + denseIndex), 1);

    if (previous == 0)
    {
        auto sparseVoxelIndex        = atomicAdd (filledNodeCounter, 1);
        denseToSparseLUT[denseIndex] = sparseVoxelIndex;
    }
}
} // namespace chunking


namespace Kernel {

template <typename... Arguments>
void pointCounting (const KernelConfig& config, Arguments&&... args)
{
    if (config.cloudType == CLOUD_FLOAT_UINT8_T)
    {
        executeKernel (
                chunking::kernelPointCounting<float>,
                config.threadAmount,
                config.name,
                std::forward<Arguments> (args)...);
    }
    else
    {
        executeKernel (
                chunking::kernelPointCounting<double>,
                config.threadAmount,
                config.name,
                std::forward<Arguments> (args)...);
    }
}
} // namespace Kernel
