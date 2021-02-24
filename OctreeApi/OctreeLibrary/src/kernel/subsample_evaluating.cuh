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

namespace subsampling {


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
        uint32_t* pointsPerSubsample,
        uint32_t linearIdx,
        KernelStructs::Cloud cloud,
        KernelStructs::Gridding gridding,
        Chunk* octree)
{
    int index               = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
    SubsampleConfig* config = (SubsampleConfig*)(&subsampleSet);
    int gridIndex           = blockIdx.z;
    bool isParent           = config[gridIndex].isParent;
    int childIdx            = config[gridIndex].sparseIdx;
    if (childIdx == -1 || (isParent && index >= pointsPerSubsample[config[gridIndex].linearIdx]) ||
        (!isParent && index >= octree[childIdx].pointCount))
    {
        return;
    }

    // Access child node data
    uint32_t* childDataLUT     = config[gridIndex].lutAdress;
    uint32_t childDataLUTStart = config[gridIndex].lutStartIndex;
    uint64_t* childAveraging   = config[gridIndex].averagingAdress;

    // Get the coordinates & colors from the point within the point cloud
    uint8_t* targetCloudByte       = cloud.raw + childDataLUT[childDataLUTStart + index] * cloud.dataStride;
    Vector3<coordinateType>* point = reinterpret_cast<Vector3<coordinateType>*> (targetCloudByte);
    Vector3<colorType>* color = reinterpret_cast<Vector3<colorType>*> (targetCloudByte + sizeof (coordinateType) * 3);

    // Calculate cell index
    auto denseVoxelIndex = mapPointToGrid<coordinateType> (point, gridding);

    // Increase the point counter for the cell
    uint32_t old = atomicAdd ((countingGrid + denseVoxelIndex), 1);

    // Accumulate color information
    bool hasAveragingData   = (childAveraging != nullptr);
    uint64_t* averagingData = childAveraging + index;

    uint64_t encoded = hasAveragingData
                               ? *averagingData
                               : (static_cast<uint64_t> (color->x) << 46) | (static_cast<uint64_t> (color->y) << 28) |
                                         static_cast<uint64_t> (color->z) << 10 | static_cast<uint64_t> (1);
    atomicAdd (&(averagingGrid[denseVoxelIndex]), encoded);

    // If the thread handles the first point in a cell: increase the pointsPerSubsample and retrieve / store the sparse
    // index for the appropriate dense cell
    if (old == 0)
    {
        denseToSparseLUT[denseVoxelIndex] = atomicAdd ((pointsPerSubsample + linearIdx), 1);
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
    return 0;
#endif
}

} // namespace Kernel
