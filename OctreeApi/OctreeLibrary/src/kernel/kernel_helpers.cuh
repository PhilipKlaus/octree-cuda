#pragma once

#include "kernel_executor.cuh"
#include "kernel_structs.cuh"
#include <cstdint>
#include <cuda_runtime_api.h>


// See OctreeConverter : chunker_countsort_laszip.cpp :131
template <typename coordinateType>
inline __device__ uint32_t
        mapPointToGrid (const Vector3<coordinateType>* point, const KernelStructs::Gridding& gridding)
{
    double t  = gridding.bbSize / gridding.gridSize;
    double uX = (point->x - gridding.bbMin.x) / t;
    double uY = (point->y - gridding.bbMin.y) / t;
    double uZ = (point->z - gridding.bbMin.z) / t;

    t           = gridding.gridSize - 1.0;
    uint64_t ix = static_cast<int64_t> (fmin (uX, t));
    uint64_t iy = static_cast<int64_t> (fmin (uY, t));
    uint64_t iz = static_cast<int64_t> (fmin (uZ, t));

    return static_cast<uint32_t> (ix + iy * gridding.gridSize + iz * gridding.gridSize * gridding.gridSize);
}


/**
 * Encodes a color vector in a single uint64.
 * @tparam colorType The datatype of the point colors.
 * @param color The color vector to be encoded.
 * @return The encoded color information.
 */
template <typename colorType>
inline __device__ uint64_t encodeColors (Vector3<colorType>* color)
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
inline __device__ uint64_t encodeColors (uint16_t r, uint16_t g, uint16_t b)
{
    return (static_cast<uint64_t> (r) << 46) | (static_cast<uint64_t> (g) << 28) | static_cast<uint64_t> (b) << 10 |
           static_cast<uint64_t> (1);
}

namespace subsampling {
/**
 * Calculates the data position (index) inside the output buffer for a given node.
 *
 * @tparam coordinateType The datatype of the point coordinates
 * @tparam colorType The datatype of the point colors
 * @param octree The octree data structure
 * @param node The node for which the data position should be calculated
 * @param lastNode The previous subsampled node
 * @param leafOffset The data position of the first subsampled parent node (after all leaf nodes)
 */
template <typename coordinateType, typename colorType>
__global__ void kernelCalcNodeByteOffset (Node* octree, uint32_t node, int lastNode, const uint32_t* leafOffset)
{
    unsigned int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
    if (index > 0)
    {
        return;
    }

    octree[node].dataIdx = (lastNode == -1) ? leafOffset[0] : octree[lastNode].dataIdx + octree[lastNode].pointCount;
}
} // namespace subsampling

namespace Kernel {
template <typename... Arguments>
void calcNodeByteOffset (const Kernel::KernelConfig& config, Arguments&&... args)
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
#ifdef ERROR_CHECKS
    cudaDeviceSynchronize ();
#endif
    gpuErrchk (cudaGetLastError ());
}
} // namespace Kernel
