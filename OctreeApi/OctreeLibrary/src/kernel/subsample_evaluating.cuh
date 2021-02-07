#pragma once

#include "kernel_executor.cuh"
#include "octree_metadata.h"
#include "tools.cuh"
#include "types.cuh"


namespace subsampling {

template <typename coordinateType>
__global__ void kernelEvaluateSubsamples (
        uint8_t* cloud,
        SubsampleSet subsampleSet,
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

    SubsampleConfig* config = (SubsampleConfig*)(&subsampleSet);

    for (int i = 0; i < 8; ++i)
    {
        if (index < config[i].pointOffsetUpper)
        {
            childDataLUT      = config[i].lutAdress;
            childDataLUTStart = config[i].lutStartIndex;
            index -= config[i].pointOffsetLower;
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


namespace Kernel {

template <typename... Arguments>
float evaluateSubsamples (KernelConfig config, Arguments&&... args)
{
    if (config.cloudType == CLOUD_FLOAT_UINT8_T)
    {
        return executeKernel (
                subsampling::kernelEvaluateSubsamples<float>, config.threadAmount, std::forward<Arguments> (args)...);
    }
    else
    {
        return executeKernel (
                subsampling::kernelEvaluateSubsamples<double>, config.threadAmount, std::forward<Arguments> (args)...);
    }
}

} // namespace Kernel
