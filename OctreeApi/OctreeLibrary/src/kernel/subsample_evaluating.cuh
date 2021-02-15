#pragma once

#include "kernel_executor.cuh"
#include "kernel_helpers.cuh"
#include "kernel_structs.cuh"
#include "octree_metadata.h"
#include "tools.cuh"
#include "types.cuh"

namespace subsampling {

template <typename coordinateType, typename colorType>
__global__ void kernelEvaluateSubsamples (
        SubsampleSet test,
        uint32_t* densePointCount,
        Averaging* averagingGrid,
        int* denseToSparseLUT,
        uint32_t* sparseIndexCounter,
        KernelStructs::Cloud cloud,
        KernelStructs::Gridding gridding)
{
    int index               = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
    SubsampleConfig* config = (SubsampleConfig*)(&test);
    int gridIndex           = blockIdx.z;

    if (index >= config[gridIndex].pointAmount)
    {
        return;
    }

    // Determine child index and pick appropriate LUT data
    uint32_t* childDataLUT     = config[gridIndex].lutAdress;
    uint32_t childDataLUTStart = config[gridIndex].lutStartIndex;
    Averaging* childAveraging  = config[gridIndex].averagingAdress;

    // Get the coordinates & colors from the point within the point cloud
    uint8_t* targetCloudByte       = cloud.raw + childDataLUT[childDataLUTStart + index] * cloud.dataStride;
    Vector3<coordinateType>* point = reinterpret_cast<Vector3<coordinateType>*> (targetCloudByte);
    Vector3<colorType>* color = reinterpret_cast<Vector3<colorType>*> (targetCloudByte + sizeof (coordinateType) * 3);

    // Calculate cell index
    auto denseVoxelIndex = mapPointToGrid<coordinateType> (point, gridding);

    uint32_t old = atomicAdd ((densePointCount + denseVoxelIndex), 1);

    bool hasAveragingData    = (childAveraging != nullptr);
    Averaging* averagingData = childAveraging + index;
    atomicAdd (&(averagingGrid[denseVoxelIndex].pointCount), hasAveragingData ? averagingData->pointCount : 1);
    atomicAdd (&(averagingGrid[denseVoxelIndex].r), hasAveragingData ? averagingData->r : color->x);
    atomicAdd (&(averagingGrid[denseVoxelIndex].g), hasAveragingData ? averagingData->g : color->y);
    atomicAdd (&(averagingGrid[denseVoxelIndex].b), hasAveragingData ? averagingData->b : color->z);

    if (old == 0)
    {
        denseToSparseLUT[denseVoxelIndex] = atomicAdd (sparseIndexCounter, 1);
    }
}
} // namespace subsampling


namespace Kernel {

template <typename... Arguments>
float evaluateSubsamples (KernelConfig config, Arguments&&... args)
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
        subsampling::kernelEvaluateSubsamples<float, uint8_t><<<grid, block>>> (std::forward<Arguments> (args)...);
        timer.stop ();
        gpuErrchk (cudaGetLastError ());
        return timer.getMilliseconds ();
    }
    else
    {
        tools::KernelTimer timer;
        timer.start ();
        subsampling::kernelEvaluateSubsamples<double, uint8_t><<<grid, block>>> (std::forward<Arguments> (args)...);
        timer.stop ();
        gpuErrchk (cudaGetLastError ());
        return timer.getMilliseconds ();
    }
}

} // namespace Kernel
