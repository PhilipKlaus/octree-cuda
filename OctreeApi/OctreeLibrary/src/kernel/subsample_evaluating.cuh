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

/**
 * Encodes a color vector in a single uint64.
 * @tparam colorType The datatype of the point colors.
 * @param color The color vector to be encoded.
 * @return The encoded color information.
 */
template <typename colorType>
__device__ uint64_t encodeColors (Vector3<colorType>* color)
{
    return (static_cast<uint64_t> (color->x) << 46) | (static_cast<uint64_t> (color->y) << 28) |
           static_cast<uint64_t> (color->z) << 10 | static_cast<uint64_t> (1);
}

/**
 * Encodes a color vector in a single uint64.
 * @tparam colorType The datatype of the point colors.
 * @param color The color vector to be encoded.
 * @return The encoded color information.
 */
 /**
  * Encodes three color components (r,g,b) in a single uint64.
  * @param r The red color component.
  * @param g The green color component.
  * @param b The blue color component.
  * @return The encoded color information.
  */
__device__ uint64_t encodeColors (uint16_t r, uint16_t g, uint16_t b)
{
    return (static_cast<uint64_t> (r) << 46) | (static_cast<uint64_t> (g) << 28) | static_cast<uint64_t> (b) << 10 |
           static_cast<uint64_t> (1);
}


/**
 * Places a 3-dimensional grid over 8 octree children nodes and maps their points to a target cell.
 * The kernel counts how many points fall into each cell of the counting grid.
 * Furthermore all color information from all points in a cell are summed up (needed for color averaging).
 * The amount of occupied cells is stored as pointCount in the parent node (octree).
 *
 * @tparam coordinateType The datatype of the 3D coordinates.
 * @tparam colorType The datatype of the point colors.
 * @param outputBuffer The binary output buffer.
 * @param countingGrid A 3-dimensional grid which stores the amount of points per node.
 * @param octree The octree data structure.
 * @param averagingGrid A 3-dimensional grid which stores averaged color information per node.
 * @param denseToSparseLUT Maps dense to sparse indices.
 * @param lut Stores the point indices of all points within a node.
 * @param cloud The point cloud data.
 * @param gridding Contains gridding related data.
 * @param nodeIdx The actual parent (target) node.
 */
template <typename coordinateType, typename colorType>
__global__ void kernelEvaluateSubsamples (
        OutputBuffer* outputBuffer,
        uint32_t* countingGrid,
        Node* octree,
        uint64_t* averagingGrid,
        int* denseToSparseLUT,
        PointLut* lut,
        KernelStructs::Cloud cloud,
        KernelStructs::Gridding gridding,
        uint32_t nodeIdx)
{
    unsigned int localPointIdx = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

    int childIdx = octree[nodeIdx].childNodes[blockIdx.z];
    auto* child  = octree + childIdx;

    if (childIdx == -1 || localPointIdx >= child->pointCount)
    {
        return;
    }

    // Get pointer to the output data entry
    PointLut* src = lut + child->dataIdx + localPointIdx;

    // Get the coordinates & colors from the point within the point cloud
    uint8_t* srcCloudByte   = cloud.raw + (*src) * cloud.dataStride;
    auto* point             = reinterpret_cast<Vector3<coordinateType>*> (srcCloudByte);
    OutputBuffer* srcBuffer = outputBuffer + child->dataIdx + localPointIdx;

    // Calculate cell index
    auto denseVoxelIndex = mapPointToGrid<coordinateType> (point, gridding);

    // Increase the point counter for the cell
    uint32_t old = atomicAdd ((countingGrid + denseVoxelIndex), 1);

    // Encode the point color and add it up
    uint64_t encoded = encodeColors (srcBuffer->r, srcBuffer->g, srcBuffer->b);

    atomicAdd (&(averagingGrid[denseVoxelIndex]), encoded);

    // If the thread handles the first point in a cell:
    // - increase the point count in the parent cell
    // - Add dense-to-sparse-lut entry
    if (old == 0)
    {
        denseToSparseLUT[denseVoxelIndex] = atomicAdd (&(octree[nodeIdx].pointCount), 1);
    }
}
} // namespace subsampling


namespace Kernel {

template <typename... Arguments>
void evaluateSubsamples (const KernelConfig& config, Arguments&&... args)
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
        subsampling::kernelEvaluateSubsamples<float, uint8_t><<<grid, block>>> (std::forward<Arguments> (args)...);
    }
    else
    {
        subsampling::kernelEvaluateSubsamples<double, uint8_t><<<grid, block>>> (std::forward<Arguments> (args)...);
    }
#ifdef KERNEL_TIMINGS
    timer.stop ();
    Timing::TimeTracker::getInstance ().trackKernelTime (timer, config.name);
#endif
    gpuErrchk (cudaGetLastError ());
}

} // namespace Kernel
