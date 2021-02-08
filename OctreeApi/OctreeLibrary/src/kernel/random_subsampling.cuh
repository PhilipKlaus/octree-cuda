#pragma once

#include "kernel_executor.cuh"
#include "octree_metadata.h"
#include "tools.cuh"
#include "types.cuh"

namespace subsampling {

template <typename coordinateType, typename colorType>
__global__ void kernelPerformAveraging (
        uint8_t* cloud,
        SubsampleSet subsampleSet,
        Averaging* parentAveragingData,
        int* denseToSparseLUT,
        PointCloudMetadata metadata,
        uint32_t gridSideLength,
        uint32_t accumulatedPoints)
{
    int thread = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
    if (thread >= accumulatedPoints)
    {
        return;
    }

    // Determine child index and pick appropriate LUT data
    uint32_t* childDataLUT     = nullptr;
    Averaging* childAveraging  = nullptr;
    uint32_t childDataLUTStart = 0;

    SubsampleConfig* config = reinterpret_cast<SubsampleConfig*> (&subsampleSet);

    for (uint8_t i = 0; i < 8; ++i)
    {
        if (thread < config[i].pointOffsetUpper)
        {
            childDataLUT      = config[i].lutAdress;
            childAveraging    = config[i].averagingAdress;
            childDataLUTStart = config[i].lutStartIndex;
            thread -= config[i].pointOffsetLower;
            break;
        }
    }

    uint8_t* targetCloudByte = cloud + childDataLUT[childDataLUTStart + thread] * metadata.pointDataStride;

    // Get the coordinates & colors from the point within the point cloud
    Vector3<coordinateType>* point = reinterpret_cast<Vector3<coordinateType>*> (targetCloudByte);
    Vector3<colorType>* color = reinterpret_cast<Vector3<colorType>*> (targetCloudByte + sizeof (coordinateType) * 3);

    // 1. Calculate the index within the dense grid of the evaluateSubsamples
    auto denseVoxelIndex = tools::calculateGridIndex (point, metadata, gridSideLength);

    int sparseIndex = denseToSparseLUT[denseVoxelIndex];

    bool hasAveragingData = (childAveraging != nullptr);

    Averaging* averagingData = childAveraging + thread;
    atomicAdd (&(parentAveragingData[sparseIndex].pointCount), hasAveragingData ? averagingData->pointCount : 1);
    atomicAdd (&(parentAveragingData[sparseIndex].r), hasAveragingData ? averagingData->r : color->x);
    atomicAdd (&(parentAveragingData[sparseIndex].g), hasAveragingData ? averagingData->g : color->y);
    atomicAdd (&(parentAveragingData[sparseIndex].b), hasAveragingData ? averagingData->b : color->z);
}


// Move point indices from old (child LUT) to new (parent LUT)
template <typename coordinateType>
__global__ void kernelRandomPointSubsample (
        uint8_t* cloud,
        SubsampleSet subsampleSet,
        uint32_t* parentDataLUT,
        uint32_t* countingGrid,
        int* denseToSparseLUT,
        uint32_t* sparseIndexCounter,
        PointCloudMetadata metadata,
        uint32_t gridSideLength,
        uint32_t* randomIndices,
        uint32_t accumulatedPoints,
        bool replacementScheme)
{
    int thread = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
    if (thread >= accumulatedPoints)
    {
        return;
    }

    // Determine child index and pick appropriate LUT data
    uint32_t* childDataLUT     = nullptr;
    uint32_t childDataLUTStart = 0;

    SubsampleConfig* config = (SubsampleConfig*)(&subsampleSet);

    for (uint8_t i = 0; i < 8; ++i)
    {
        if (thread < config[i].pointOffsetUpper)
        {
            childDataLUT      = config[i].lutAdress;
            childDataLUTStart = config[i].lutStartIndex;
            thread -= config[i].pointOffsetLower;
            break;
        }
    }

    uint32_t lutItem = childDataLUT[childDataLUTStart + thread];

    // Get the point within the point cloud
    Vector3<coordinateType>* point =
            reinterpret_cast<Vector3<coordinateType>*> (cloud + lutItem * metadata.pointDataStride);

    // 1. Calculate the index within the dense grid of the evaluateSubsamples
    auto denseVoxelIndex = tools::calculateGridIndex (point, metadata, gridSideLength);

    int sparseIndex = denseToSparseLUT[denseVoxelIndex];

    // 2. We are only interested in the last point within a node -> Implicitly reset the countingGrid
    auto oldIndex = atomicSub ((countingGrid + denseVoxelIndex), 1);

    if (sparseIndex == -1 || oldIndex != randomIndices[sparseIndex])
    {
        return;
    }

    // Move subsampled point to parent
    parentDataLUT[sparseIndex] = lutItem;
    childDataLUT[childDataLUTStart + thread] =
            replacementScheme ? childDataLUT[childDataLUTStart + thread] : INVALID_INDEX;

    // Reset all subsampling data data
    denseToSparseLUT[denseVoxelIndex] = -1;
    *sparseIndexCounter               = 0;
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
        Averaging* averagingData,
        uint32_t* countingGrid,
        uint32_t gridNodes)
{
    int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

    if (index >= gridNodes)
    {
        return;
    }

    int sparseIndex = denseToSparseLUT[index];

    if (sparseIndex == -1)
    {
        return;
    }

    // Generate random value for point picking
    randomIndices[sparseIndex] =
            static_cast<uint32_t> (ceil (curand_uniform (&states[threadIdx.x]) * countingGrid[index]));

    // Reset Averaging data
    averagingData[sparseIndex].r          = 0.f;
    averagingData[sparseIndex].g          = 0.f;
    averagingData[sparseIndex].b          = 0.f;
    averagingData[sparseIndex].pointCount = 0;
}
} // namespace subsampling

namespace Kernel {

template <typename... Arguments>
float performAveraging (KernelConfig config, Arguments&&... args)
{
    if (config.cloudType == CLOUD_FLOAT_UINT8_T)
    {
        return executeKernel (
                subsampling::kernelPerformAveraging<float, uint8_t>,
                config.threadAmount,
                std::forward<Arguments> (args)...);
    }
    else
    {
        return executeKernel (
                subsampling::kernelPerformAveraging<double, uint8_t>,
                config.threadAmount,
                std::forward<Arguments> (args)...);
    }
}

template <typename... Arguments>
float randomPointSubsampling (KernelConfig config, Arguments&&... args)
{
    if (config.cloudType == CLOUD_FLOAT_UINT8_T)
    {
        return executeKernel (
                subsampling::kernelRandomPointSubsample<float>, config.threadAmount, std::forward<Arguments> (args)...);
    }
    else
    {
        return executeKernel (
                subsampling::kernelRandomPointSubsample<double>,
                config.threadAmount,
                std::forward<Arguments> (args)...);
    }
}
} // namespace Kernel