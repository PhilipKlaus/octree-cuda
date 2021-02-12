#pragma once

#include "kernel_executor.cuh"
#include "octree_metadata.h"
#include "tools.cuh"
#include "types.cuh"
#include "kernel_helpers.cuh"
#include "kernel_structs.cuh"

namespace subsampling {

template <typename coordinateType>
__global__ void kernelEvaluateSubsamples (
        SubsampleSetTest test,
        uint32_t* densePointCount,
        KernelStructs::Cloud cloud,
        KernelStructs::Gridding gridding)
{
    int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
    SubsampleConfigTest* config = (SubsampleConfigTest*)(&test);
    int gridIndex = blockIdx.z;

    if (index >= config[gridIndex].pointAmount)
    {
        return;
    }

    // Determine child index and pick appropriate LUT data
    uint32_t* childDataLUT      = config[gridIndex].lutAdress;
    uint32_t childDataLUTStart = config[gridIndex].lutStartIndex;

    Vector3<coordinateType>* point = reinterpret_cast<Vector3<coordinateType>*> (
            cloud.raw + childDataLUT[childDataLUTStart + index] * cloud.dataStride);

    // Calculate cell index
    auto denseVoxelIndex = mapPointToGrid<coordinateType> (point, gridding);

    // Increase point count in cell
    auto oldIndex = atomicAdd ((densePointCount + denseVoxelIndex), 1);
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
        // Calculate kernel dimensions
        dim3 grid, block;

        auto blocks = ceil (static_cast<double> (config.threadAmount) / BLOCK_SIZE_MAX);
        auto gridX  = blocks < GRID_SIZE_MAX ? blocks : GRID_SIZE_MAX;
        auto gridY  = ceil (blocks / GRID_SIZE_MAX);

        block = dim3 (BLOCK_SIZE_MAX, 1, 1);
        grid  = dim3 (static_cast<unsigned int> (gridX), static_cast<unsigned int> (gridY), 8);

        tools::KernelTimer timer;
        timer.start ();
        subsampling::kernelEvaluateSubsamples<double><<<grid, block>>> (std::forward<Arguments> (args)...);
        timer.stop ();
        gpuErrchk (cudaGetLastError ());
        return timer.getMilliseconds ();
    }
}

} // namespace Kernel
