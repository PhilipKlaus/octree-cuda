#pragma once

#include "octree_metadata.h"
#include "types.cuh"
#include "tools.cuh"


namespace subsampling {

template <typename coordinateType, typename colorType>
__global__ void kernelPerformAveraging (
        uint8_t* cloud,
        SubsampleConfig* subsampleData,
        Averaging* parentAveragingData,
        int* denseToSparseLUT,
        PointCloudMetadata<coordinateType> metadata,
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
    Averaging* childAveraging  = nullptr;
    uint32_t childDataLUTStart = 0;

    for (int i = 0; i < 8; ++i)
    {
        if (index < subsampleData[i].pointOffsetUpper)
        {
            childDataLUT      = subsampleData[i].lutAdress;
            childAveraging    = subsampleData[i].averagingAdress;
            childDataLUTStart = subsampleData[i].lutStartIndex;
            index -= subsampleData[i].pointOffsetLower;
            break;
        }
    }

    // Get the coordinates from the point within the point cloud
    Vector3<coordinateType>* point = reinterpret_cast<Vector3<coordinateType>*> (
            cloud + childDataLUT[childDataLUTStart + index] * metadata.pointDataStride);

    // Get the color from the point within the point cloud
    Vector3<colorType>* color = reinterpret_cast<Vector3<colorType>*> (
            cloud + childDataLUT[childDataLUTStart + index] * metadata.pointDataStride + sizeof (coordinateType) * 3);

    // 1. Calculate the index within the dense grid of the evaluateSubsamples
    auto denseVoxelIndex = tools::calculateGridIndex (point, metadata, gridSideLength);

    int sparseIndex = denseToSparseLUT[denseVoxelIndex];

    bool hasAveragingData = (childAveraging != nullptr);
    atomicAdd (&(parentAveragingData[sparseIndex].pointCount), hasAveragingData ? childAveraging[index].pointCount : 1);

    atomicAdd (&(parentAveragingData[sparseIndex].r), hasAveragingData ? childAveraging[index].r : color->x);
    atomicAdd (&(parentAveragingData[sparseIndex].g), hasAveragingData ? childAveraging[index].g : color->y);
    atomicAdd (&(parentAveragingData[sparseIndex].b), hasAveragingData ? childAveraging[index].b : color->z);
}


// Move point indices from old (child LUT) to new (parent LUT)
template <typename coordinateType>
__global__ void kernelRandomPointSubsample (
        uint8_t* cloud,
        SubsampleConfig* subsampleData,
        uint32_t* parentDataLUT,
        Averaging* averagingData,
        uint32_t* countingGrid,
        int* denseToSparseLUT,
        uint32_t* sparseIndexCounter,
        PointCloudMetadata<coordinateType> metadata,
        uint32_t gridSideLength,
        uint32_t* randomIndices,
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

    // Get the point within the point cloud
    Vector3<coordinateType>* point = reinterpret_cast<Vector3<coordinateType>*> (
            cloud + childDataLUT[childDataLUTStart + index] * metadata.pointDataStride);

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
    parentDataLUT[sparseIndex] = childDataLUT[childDataLUTStart + index];
    //childDataLUT[childDataLUTStart + index] = INVALID_INDEX; // Additive strategy

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