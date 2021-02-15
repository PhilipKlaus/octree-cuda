#pragma once

#include "kernel_executor.cuh"
#include "octree_metadata.h"
#include "tools.cuh"
#include "types.cuh"
#include "kernel_helpers.cuh"
#include "kernel_structs.cuh"

namespace subsampling {

// Move point indices from old (child LUT) to new (parent LUT
template <typename coordinateType>
__global__ void kernelRandomPointSubsample (
        SubsampleSet test,
        uint32_t* parentDataLUT,
        Averaging* parentAveraging,
        uint32_t* countingGrid,
        Averaging* averagingGrid,
        int* denseToSparseLUT,
        uint32_t* sparseIndexCounter,
        KernelStructs::Cloud cloud,
        KernelStructs::Gridding gridding,
        uint32_t* randomIndices,
        bool replacementScheme)
{
    int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
    SubsampleConfig* config = (SubsampleConfig*)(&test);
    int gridIndex = blockIdx.z;

    if (index >= config[gridIndex].pointAmount)
    {
        return;
    }
    // Determine child index and pick appropriate LUT data
    uint32_t* childDataLUT     = config[gridIndex].lutAdress;
    uint32_t childDataLUTStart = config[gridIndex].lutStartIndex;

    uint32_t lutItem = childDataLUT[childDataLUTStart + index];

    // Get the point within the point cloud
    Vector3<coordinateType>* point =
            reinterpret_cast<Vector3<coordinateType>*> (cloud.raw + lutItem * cloud.dataStride);

    // Calculate cell index
    auto denseVoxelIndex = mapPointToGrid<coordinateType> (point, gridding);

    int sparseIndex = denseToSparseLUT[denseVoxelIndex];

    // 2. We are only interested in the last point within a node -> Implicitly reset the countingGrid
    auto oldIndex = atomicSub ((countingGrid + denseVoxelIndex), 1);

    if (sparseIndex == -1 || oldIndex != randomIndices[sparseIndex])
    {
        return;
    }

    // Move subsampled point to parent
    parentDataLUT[sparseIndex] = lutItem;
    parentAveraging[sparseIndex] = averagingGrid[denseVoxelIndex];
    childDataLUT[childDataLUTStart + index] =
            replacementScheme ? childDataLUT[childDataLUTStart + index] : INVALID_INDEX;

    // Reset all subsampling data data
    denseToSparseLUT[denseVoxelIndex] = -1;
    *sparseIndexCounter               = 0;
    averagingGrid[denseVoxelIndex].pointCount = 0;
    averagingGrid[denseVoxelIndex].r = 0;
    averagingGrid[denseVoxelIndex].g = 0;
    averagingGrid[denseVoxelIndex].b = 0;
}

// http://ianfinlayson.net/class/cpsc425/notes/cuda-random
__global__ void kernelInitRandoms (unsigned int seed, curandState_t* states, uint32_t nodeAmount)
{
    int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

    if (index >= nodeAmount)
    {
        return;
    }

    curand_init (seed, index, 0, &states[index]);
}

__global__ void kernelGenerateRandoms (
        curandState_t* states,
        uint32_t* randomIndices,
        int* denseToSparseLUT,
        uint32_t* sparseIndexCounter,
        uint32_t* countingGrid,
        uint32_t gridNodes)
{
    int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

    bool cellIsEmpty = countingGrid[index] == 0;

    if (index >= gridNodes || cellIsEmpty)
    {
        return;
    }

    uint32_t sparseIndex = denseToSparseLUT[index];

    randomIndices[sparseIndex] = static_cast<uint32_t> (ceil (curand_uniform (&states[threadIdx.x]) * countingGrid[index]));
}
} // namespace subsampling

namespace Kernel {

template <typename... Arguments>
float randomPointSubsampling (KernelConfig config, Arguments&&... args)
{
    // Calculate kernel dimensions
    dim3 grid, block;

    auto blocks = ceil (static_cast<double> (config.threadAmount) / 128);
    auto gridX  = blocks < GRID_SIZE_MAX ? blocks : GRID_SIZE_MAX;
    auto gridY  = ceil (blocks / GRID_SIZE_MAX);

    block = dim3 (128, 1, 1);
    grid  = dim3 (static_cast<unsigned int> (gridX), static_cast<unsigned int> (gridY), 8);


    if (config.cloudType == CLOUD_FLOAT_UINT8_T)
    {
        tools::KernelTimer timer;
        timer.start ();
        subsampling::kernelRandomPointSubsample<float><<<grid, block>>> (std::forward<Arguments> (args)...);
        timer.stop ();
        gpuErrchk (cudaGetLastError ());
        return timer.getMilliseconds ();
    }
    else
    {
        tools::KernelTimer timer;
        timer.start ();
        subsampling::kernelRandomPointSubsample<double><<<grid, block>>> (std::forward<Arguments> (args)...);
        timer.stop ();
        gpuErrchk (cudaGetLastError ());
        return timer.getMilliseconds ();
    }
}
} // namespace Kernel