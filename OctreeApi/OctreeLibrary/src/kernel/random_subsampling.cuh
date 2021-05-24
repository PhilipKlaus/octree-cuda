/**
 * @file random_subsampling.cuh
 * @author Philip Klaus
 * @brief Contains code for a randomized subampling of points within 8 child nodes.
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
 * Generates one random number, depending on the point amount in a node.
 * Important! This kernel does not generate a random point index, instead it generates
 * a random number between [0 <-> points-in-a-node]. The actual assignment from this
 * generated random number to a 3D point is performed in kernelRandomPointSubsample.
 *
 * @param states The initialized random states.
 * @param randomIndices An array for storing one randomly generated number per filled cell.
 * @param denseToSparseLUT Maps dense to sparse indices.
 * @param countingGrid Holds the amount of points per cell.
 * @param cellAmount The amount of cells for which the kernel is called.
 */
__global__ void kernelGenerateRandoms (
        curandState_t* states,
        uint32_t* randomIndices,
        const int* denseToSparseLUT,
        const uint32_t* countingGrid,
        uint32_t cellAmount)
{
    unsigned int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

    bool cellIsEmpty = countingGrid[index] == 0;

    if (index >= cellAmount || cellIsEmpty)
    {
        return;
    }

    uint32_t sparseIndex = denseToSparseLUT[index];

    randomIndices[sparseIndex] =
            static_cast<uint32_t> (ceil (curand_uniform (&states[threadIdx.x]) * countingGrid[index]));
}

/**
 * Picks a random point from a child node and stores the point index in a point LUT.
 * Further, the point data (coordinates, colors) are written to the binary output buffer.
 * After processing the kernel, resets all temporary needed data structures.
 *
 * @tparam coordinateType The datatype of the 3D coordinates.
 * @tparam colorType The datatype of the point colors.
 * @param outputBuffer The binary output buffer.
 * @param countingGrid A 3-dimensional grid which stores the amount of points per node.
 * @param averagingGrid A 3-dimensional grid which stores averaged color information per node.
 * @param denseToSparseLUT Maps dense to sparse indices.
 * @param cloud The point cloud data.
 * @param gridding Contains gridding related data.
 * @param randomIndices The generated random indices for each subsampled node.
 * @param lut Stores the point indices of all points within a node.
 * @param octree The octree data structure.
 * @param nodeIdx The actual parent (target) node.
 */
template <typename coordinateType, typename colorType>
__global__ void kernelRandomPointSubsample (
        OutputBuffer* outputBuffer,
        uint32_t* countingGrid,
        uint64_t* averagingGrid,
        int* denseToSparseLUT,
        KernelStructs::Cloud cloud,
        KernelStructs::Gridding gridding,
        BoundingBox cubic,
        uint32_t* randomIndices,
        PointLut* lut,
        Node* octree,
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

    // Get the point within the point cloud
    uint8_t* srcPoint = cloud.raw + (*src) * cloud.dataStride;
    auto* point       = reinterpret_cast<Vector3<coordinateType>*> (srcPoint);

    // Calculate the dense and sparse cell index
    auto denseVoxelIndex = mapPointToGrid<coordinateType> (point, gridding);
    int sparseIndex      = denseToSparseLUT[denseVoxelIndex];

    // Decrease the point counter per cell
    auto oldIndex = atomicSub ((countingGrid + denseVoxelIndex), 1);

    // If the actual thread does not handle the randomly chosen point, exit now.
    if (sparseIndex == -1 || oldIndex != randomIndices[sparseIndex])
    {
        return;
    }

    // Move subsampled averaging and point-LUT data to parent node
    PointLut* dst = lut + octree[nodeIdx].dataIdx + sparseIndex;
    *dst          = *src;

    uint64_t encoded = averagingGrid[denseVoxelIndex];
    auto amount      = static_cast<uint16_t> (encoded & 0x3FF);

    uint64_t decoded[3] = {
            ((encoded >> 46) & 0xFFFF) / amount,
            ((encoded >> 28) & 0xFFFF) / amount,
            ((encoded >> 10) & 0xFFFF) / amount};

    // Write coordinates and colors to output buffer
    OutputBuffer* out = outputBuffer + octree[nodeIdx].dataIdx + sparseIndex;
    out->x            = static_cast<int32_t> (floor ((point->x - cubic.min.x) * cloud.scaleFactor.x));
    out->y            = static_cast<int32_t> (floor ((point->y - cubic.min.y) * cloud.scaleFactor.y));
    out->z            = static_cast<int32_t> (floor ((point->z - cubic.min.z) * cloud.scaleFactor.z));
    out->r            = static_cast<uint16_t> (decoded[0]);
    out->g            = static_cast<uint16_t> (decoded[1]);
    out->b            = static_cast<uint16_t> (decoded[2]);

    // Reset all temporary data structures
    denseToSparseLUT[denseVoxelIndex] = -1;
    averagingGrid[denseVoxelIndex]    = 0;
}

/**
 * Pick the last point within a specific subsampling grid cell and stores the point index in a point LUT.
 * Further, the point data (coordinates, colors) are written to the binary output buffer.
 * After processing the kernel, resets all temporary needed data structures.
 *
 * @tparam coordinateType The datatype of the 3D coordinates.
 * @tparam colorType The datatype of the point colors.
 * @param outputBuffer The binary output buffer.
 * @param countingGrid A 3-dimensional grid which stores the amount of points per node.
 * @param averagingGrid A 3-dimensional grid which stores averaged color information per node.
 * @param denseToSparseLUT Maps dense to sparse indices.
 * @param cloud The point cloud data.
 * @param gridding Contains gridding related data.
 * @param randomIndices The generated random indices for each subsampled node.
 * @param lut Stores the point indices of all points within a node.
 * @param octree The octree data structure.
 * @param nodeIdx The actual parent (target) node.
 */
template <typename coordinateType, typename colorType>
__global__ void kernelFirstPointSubsample (
        OutputBuffer* outputBuffer,
        uint32_t* countingGrid,
        uint64_t* averagingGrid,
        int* denseToSparseLUT,
        KernelStructs::Cloud cloud,
        KernelStructs::Gridding gridding,
        BoundingBox cubic,
        PointLut* lut,
        Node* octree,
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

    // Get the point within the point cloud
    uint8_t* srcPoint = cloud.raw + (*src) * cloud.dataStride;
    auto* point       = reinterpret_cast<Vector3<coordinateType>*> (srcPoint);

    // Calculate the dense and sparse cell index
    auto denseVoxelIndex = mapPointToGrid<coordinateType> (point, gridding);
    int sparseIndex      = denseToSparseLUT[denseVoxelIndex];

    // Decrease the point counter per cell
    auto oldIndex = atomicSub ((countingGrid + denseVoxelIndex), 1);

    // If the actual thread does not handle the randomly chosen point, exit now.
    if (sparseIndex == -1 || oldIndex != 1)
    {
        return;
    }

    // Move subsampled averaging and point-LUT data to parent node
    PointLut* dst = lut + octree[nodeIdx].dataIdx + sparseIndex;
    *dst          = *src;

    uint64_t encoded = averagingGrid[denseVoxelIndex];
    auto amount      = static_cast<uint16_t> (encoded & 0x3FF);

    uint64_t decoded[3] = {
            ((encoded >> 46) & 0xFFFF) / amount,
            ((encoded >> 28) & 0xFFFF) / amount,
            ((encoded >> 10) & 0xFFFF) / amount};

    // Write coordinates and colors to output buffer
    OutputBuffer* out = outputBuffer + octree[nodeIdx].dataIdx + sparseIndex;
    out->x            = static_cast<int32_t> (floor ((point->x - cubic.min.x) * cloud.scaleFactor.x));
    out->y            = static_cast<int32_t> (floor ((point->y - cubic.min.y) * cloud.scaleFactor.y));
    out->z            = static_cast<int32_t> (floor ((point->z - cubic.min.z) * cloud.scaleFactor.z));
    out->r            = static_cast<uint16_t> (decoded[0]);
    out->g            = static_cast<uint16_t> (decoded[1]);
    out->b            = static_cast<uint16_t> (decoded[2]);

    // Reset all temporary data structures
    denseToSparseLUT[denseVoxelIndex] = -1;
    averagingGrid[denseVoxelIndex]    = 0;
}

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
void randomPointSubsampling (const KernelConfig& config, Arguments&&... args)
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
        subsampling::kernelRandomPointSubsample<float, uint8_t><<<grid, block>>> (std::forward<Arguments> (args)...);
    }
    else
    {
        subsampling::kernelRandomPointSubsample<double, uint8_t><<<grid, block>>> (std::forward<Arguments> (args)...);
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

template <typename... Arguments>
void firstPointSubsampling (const KernelConfig& config, Arguments&&... args)
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
        subsampling::kernelFirstPointSubsample<float, uint8_t><<<grid, block>>> (std::forward<Arguments> (args)...);
    }
    else
    {
        subsampling::kernelFirstPointSubsample<double, uint8_t><<<grid, block>>> (std::forward<Arguments> (args)...);
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

template <typename... Arguments>
void calcNodeByteOffset (const KernelConfig& config, Arguments&&... args)
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