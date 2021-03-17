/**
 * @file random_subsampling.cuh
 * @author Philip Klaus
 * @brief Contains code for a randomized subampling of points within 8 child nodes.
 */

#pragma once

#include "kernel_executor.cuh"
#include "kernel_helpers.cuh"
#include "kernel_structs.cuh"
#include "metadata.cuh"
#include "tools.cuh"
#include "types.cuh"

namespace subsampling {

/**
 * Generates one random number, depending on the point amount in a cell.
 * Important! This kernel does not generate a random point index, instead it generates
 * a random number between [0 <-> points-in-a-cell]. The actual assignment from this
 * generated random number to a 3D point is performed in kernelRandomPointSubsample.
 *
 * @param states The initialized random states.
 * @param randomIndices An array for storing one randomly generated number per filled cell.
 * @param denseToSparseLUT Maps dense to sparse indices.
 * @param countingGrid Holds the amount of points per cell.
 * @param cellAmount The amount of cells for which the kernel is called.
 */
__global__ void kernelGenerateRandoms (
        curandState_t* states,
        uint32_t* randomIndices,
        const int* denseToSparseLUT,
        const uint32_t* countingGrid,
        uint32_t cellAmount)
{
    int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

    bool cellIsEmpty = countingGrid[index] == 0;

    if (index >= cellAmount || cellIsEmpty)
    {
        return;
    }

    uint32_t sparseIndex = denseToSparseLUT[index];

    randomIndices[sparseIndex] =
            static_cast<uint32_t> (ceil (curand_uniform (&states[threadIdx.x]) * countingGrid[index]));
}

/**
 * Evaluates on random point per cell and assign its averaging and point-lut data to the parent node.
 * Furthermore this kernel resets all temporary needed data structures.
 *
 * @tparam coordinateType The datatype of the 3D coordinates.
 * @tparam colorType The datatype of the point colors.
 * @param parentDataLUT The point-LUT of the parent node.
 * @param parentAveraging The averaging data of the parent node.
 * @param countingGrid Holds the amount of points per cell.
 * @param averagingGrid Holds the averaging data from all 8 child nodes.
 * @param denseToSparseLUT Maps dense to sparse indices.
 * @param filledCellCounter Hlds the amount of filled cells (!=0) wihtin the counting grid.
 * @param cloud Holds the point cloud data.
 * @param gridding Holds data necessary to map a 3D point to a cell.
 * @param randomIndices Holds the previously generated random numbers for each subsampling cell.
 * @param replacementScheme Determines if the replacement scheme or the averaging scheme should be applied.
 */
template <typename coordinateType>
__global__ void kernelRandomPointSubsample (
        SubsampleSet test,
        uint32_t* countingGrid,
        uint64_t* averagingGrid,
        int* denseToSparseLUT,
        KernelStructs::Cloud cloud,
        KernelStructs::Gridding gridding,
        uint32_t* randomIndices,
        OutputData* output,
        Chunk *octree,
        uint32_t nodeIdx)
{
    int localPointIdx = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

    auto* config = (SubsampleConfig*)(&test);
    bool isParent           = config[blockIdx.z].isParent;
    int sparseIdx           = config[blockIdx.z].sparseIdx;

    if (sparseIdx == -1 || (isParent && localPointIdx >= octree[sparseIdx].pointCount) ||
        (!isParent && localPointIdx >= config[blockIdx.z].leafPointAmount))
    {
        return;
    }

    // Get pointer to the output data entry
    OutputData* src = output + octree[sparseIdx].chunkDataIndex + localPointIdx;

    // Get the point within the point cloud
    Vector3<coordinateType>* point =
            reinterpret_cast<Vector3<coordinateType>*> (cloud.raw + src->pointIdx * cloud.dataStride);

    // Calculate the dense and sparse cell index
    auto denseVoxelIndex = mapPointToGrid<coordinateType> (point, gridding);
    int sparseIndex      = denseToSparseLUT[denseVoxelIndex];

    // Decrease the point counter per cell
    auto oldIndex = atomicSub ((countingGrid + denseVoxelIndex), 1);

    // If the actual thread does not handle the randomly chosen point, exit now.
    if (sparseIndex == -1 || oldIndex != randomIndices[sparseIndex])
    {
        return;
    }

    // Move subsampled averaging and point-LUT data to parent node
    OutputData* dst = output + octree[nodeIdx].chunkDataIndex + sparseIndex;
    dst->pointIdx   = src->pointIdx;

    uint64_t encoded = averagingGrid[denseVoxelIndex];
    auto amount  = static_cast<uint16_t> (encoded & 0x3FF);

    dst->encoded = ((((encoded >> 46) & 0xFFFF) / amount) << 46) | ((((encoded >> 28) & 0xFFFF) / amount) << 28) |
                   ((((encoded >> 10) & 0xFFFF) / amount) << 10) | 1;

    // Reset all temporary data structures
    denseToSparseLUT[denseVoxelIndex] = -1;
    averagingGrid[denseVoxelIndex]    = 0;
}

template <typename coordinateType, typename colorType>
__global__ void kernelCalcNodeByteOffset (Chunk *octree, uint32_t nodeIndex, int lastNode, uint32_t *leafOffset)
{
    int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
    if (index > 0)
    {
        return;
    }

    octree[nodeIndex].chunkDataIndex = (lastNode == -1) ? leafOffset[0] : octree[lastNode].chunkDataIndex + octree[lastNode].pointCount;
}

} // namespace subsampling

namespace Kernel {

template <typename... Arguments>
void randomPointSubsampling (const KernelConfig& config, Arguments&&... args)
{
    // Calculate kernel dimensions
    dim3 grid, block;

    auto blocks = ceil (static_cast<double> (config.threadAmount) / 128);
    auto gridX  = blocks < GRID_SIZE_MAX ? blocks : GRID_SIZE_MAX;
    auto gridY  = ceil (blocks / GRID_SIZE_MAX);

    block = dim3 (128, 1, 1);
    grid  = dim3 (static_cast<unsigned int> (gridX), static_cast<unsigned int> (gridY), 8);

#ifdef KERNEL_TIMINGS
    Timing::KernelTimer timer;
    timer.start ();
#endif
    if (config.cloudType == CLOUD_FLOAT_UINT8_T)
    {
        subsampling::kernelRandomPointSubsample<float><<<grid, block>>> (std::forward<Arguments> (args)...);
    }
    else
    {
        subsampling::kernelRandomPointSubsample<double><<<grid, block>>> (std::forward<Arguments> (args)...);
    }
#ifdef KERNEL_TIMINGS
    timer.stop ();
    Timing::TimeTracker::getInstance ().trackKernelTime (timer, config.name);
#endif
    gpuErrchk (cudaGetLastError ());
}

template <typename... Arguments>
void calcNodeByteOffset (const KernelConfig& config, Arguments&&... args)
{
    auto block = dim3 (1, 1, 1);
    auto grid  = dim3 (1, 1, 1);

#ifdef KERNEL_TIMINGS
    Timing::KernelTimer timer;
    timer.start ();
#endif
    if (config.cloudType == CLOUD_FLOAT_UINT8_T)
    {
        subsampling::kernelCalcNodeByteOffset<float, uint8_t><<<grid, block>>> (std::forward<Arguments> (args)...);
    }
    else
    {
        subsampling::kernelCalcNodeByteOffset<double, uint8_t><<<grid, block>>> (std::forward<Arguments> (args)...);
    }
#ifdef KERNEL_TIMINGS
    timer.stop ();
    Timing::TimeTracker::getInstance ().trackKernelTime (timer, config.name);
#endif
    gpuErrchk (cudaGetLastError ());
}

} // namespace Kernel