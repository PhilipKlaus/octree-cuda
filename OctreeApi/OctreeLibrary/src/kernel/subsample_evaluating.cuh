/**
 * @file subsample_evaluation.cuh
 * @author Philip Klaus
 * @brief Contains code for evaluating subsample points and for summarizing color information inside subsampling cells
 */
#pragma once

#include "kernel_executor.cuh"
#include "kernel_helpers.cuh"
#include "kernel_structs.cuh"
#include "metadata.cuh"
#include "tools.cuh"
#include "types.cuh"
#include <inttypes.h>

namespace subsampling {

template <typename colorType>
__device__ uint64_t encodeColors (Vector3<colorType>* color)
{
    return (static_cast<uint64_t> (color->x) << 46) | (static_cast<uint64_t> (color->y) << 28) |
           static_cast<uint64_t> (color->z) << 10 | static_cast<uint64_t> (1);
}

/**
 * Places a 3-dimensional grid over 8 octree children nodes and maps the points inside to a target cell.
 * The kernel counts how many points fall into each cell of the counting grid.
 * Furthermore all color information from all points in a cell are summed up (needed for color averaging)
 *
 * @tparam coordinateType The datatype of the 3D coordinates.
 * @tparam colorType The datatype of the point colors.
 * @param subsampleSet Contains meta information necessary for accessing child node data.
 * @param countingGrid The actual counting grid, holding the amount of points per cell.
 * @param averagingGrid Hold the summarized color information per cell.
 * @param denseToSparseLUT Maps dense to sparse indices.
 * @param pointsPerSubsample Counts how many cells in the counting grid are actually filled.
 * @param linearIdx The linear index in pointsPerSubsample.
 * @param cloud Holds the point cloud data.
 * @param gridding Holds data necessary to map a 3D point to a cell.
 */
template <typename coordinateType, typename colorType>
__global__ void kernelEvaluateSubsamples (
        SubsampleSet subsampleSet,
        uint32_t* countingGrid,
        uint64_t* averagingGrid,
        int* denseToSparseLUT,
        OutputData* output,
        KernelStructs::NodeOutput nodeOutput,
        uint32_t parentLinearIdx,
        KernelStructs::Cloud cloud,
        KernelStructs::Gridding gridding,
        uint32_t* leafLut)
{
    int localPointIdx = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

    SubsampleConfig* config = (SubsampleConfig*)(&subsampleSet);
    bool isParent           = config[blockIdx.z].isParent;  // Is the child node a parent?
    int sparseIdx           = config[blockIdx.z].sparseIdx; // Sparse index of the child node
    int childLinearIdx      = config[blockIdx.z].linearIdx; // Is 0 if isParent = false

    if (sparseIdx == -1 || (isParent && localPointIdx >= nodeOutput.pointCount[childLinearIdx]) ||
        (!isParent && localPointIdx >= config[blockIdx.z].leafPointAmount))
    {
        return;
    }

    // Get pointer to the output data entry
    OutputData *src = output + nodeOutput.pointOffset[childLinearIdx] + localPointIdx;

    // Calculate global target point index
    uint32_t globalPointIdx = isParent ? src->pointIdx : *(leafLut + config[blockIdx.z].leafDataIdx + localPointIdx);

    // Get the coordinates & colors from the point within the point cloud
    uint8_t* targetCloudByte       = cloud.raw + globalPointIdx * cloud.dataStride;
    Vector3<coordinateType>* point = reinterpret_cast<Vector3<coordinateType>*> (targetCloudByte);
    Vector3<colorType>* color = reinterpret_cast<Vector3<colorType>*> (targetCloudByte + sizeof (coordinateType) * 3);

    // Calculate cell index
    auto denseVoxelIndex = mapPointToGrid<coordinateType> (point, gridding);

    // Increase the point counter for the cell
    uint32_t old = atomicAdd ((countingGrid + denseVoxelIndex), 1);

    // Encode the point color and add it up
    uint64_t encoded = isParent ? src->encoded : encodeColors (color);

    atomicAdd (&(averagingGrid[denseVoxelIndex]), encoded);

    // If the thread handles the first point in a cell: increase the pointsPerSubsample and retrieve / store the sparse
    // index for the appropriate dense cell
    if (old == 0)
    {
        denseToSparseLUT[denseVoxelIndex] = atomicAdd (nodeOutput.pointCount + parentLinearIdx, 1);
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

#ifdef CUDA_TIMINGS
    tools::KernelTimer timer;
    timer.start ();
    if (config.cloudType == CLOUD_FLOAT_UINT8_T)
    {
        subsampling::kernelEvaluateSubsamples<float, uint8_t><<<grid, block>>> (std::forward<Arguments> (args)...);
    }
    else
    {
        subsampling::kernelEvaluateSubsamples<double, uint8_t><<<grid, block>>> (std::forward<Arguments> (args)...);
    }
    timer.stop ();
    gpuErrchk (cudaGetLastError ());
    return timer.getMilliseconds ();
#else
    if (config.cloudType == CLOUD_FLOAT_UINT8_T)
    {
        subsampling::kernelEvaluateSubsamples<float, uint8_t><<<grid, block>>> (std::forward<Arguments> (args)...);
    }
    else
    {
        subsampling::kernelEvaluateSubsamples<double, uint8_t><<<grid, block>>> (std::forward<Arguments> (args)...);
    }
    gpuErrchk (cudaGetLastError ());
    return 0;
#endif
}

} // namespace Kernel
