#pragma once

#include "kernel_executor.cuh"

namespace subsampling {

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
 * Pick the last point within a specific subsampling grid cell and stores the point index in a point LUT.
 * Further, the point data (coordinates, colors) are written to the binary output buffer.
 * After processing the kernel, resets all temporary needed data structures.
 *
 * @tparam coordinateType The datatype of the 3D coordinates.
 * @tparam colorType The datatype of the point colors.
 * @param outputBuffer The binary output buffer.
 * @param countingGrid A 3-dimensional grid which stores the amount of points per node.
 * @param denseToSparseLUT Maps dense to sparse indices.
 * @param cloud The point cloud data.
 * @param gridding Contains gridding related data.
 * @param randomIndices The generated random indices for each subsampled node.
 * @param lut Stores the point indices of all points within a node.
 * @param octree The octree data structure.
 * @param nodeIdx The actual parent (target) node.
 */
template <typename coordinateType, typename colorType>
__global__ void kernelFirstPointSubsampleNotAveraged (
        OutputBuffer* outputBuffer,
        uint32_t* countingGrid,
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
    auto* color       = reinterpret_cast<Vector3<colorType>*> (srcPoint + 3 * sizeof (coordinateType));

    // Calculate the dense and sparse cell index
    auto denseVoxelIndex = mapPointToGrid<coordinateType> (point, gridding);

    // Decrease the point counter per cell
    auto oldIndex = atomicAdd ((countingGrid + denseVoxelIndex), 1);

    // If the actual thread does not handle the randomly chosen point, exit now.
    if (oldIndex != 0)
    {
        return;
    }

    int sparseIndex = atomicAdd (&(octree[nodeIdx].pointCount), 1);


    // Move subsampled averaging and point-LUT data to parent node
    PointLut* dst = lut + octree[nodeIdx].dataIdx + sparseIndex;
    *dst          = *src;

    // Write coordinates and colors to output buffer
    OutputBuffer* out = outputBuffer + octree[nodeIdx].dataIdx + sparseIndex;
    out->x            = static_cast<int32_t> (floor ((point->x - cubic.min.x) * cloud.scaleFactor.x));
    out->y            = static_cast<int32_t> (floor ((point->y - cubic.min.y) * cloud.scaleFactor.y));
    out->z            = static_cast<int32_t> (floor ((point->z - cubic.min.z) * cloud.scaleFactor.z));
    out->r            = static_cast<uint16_t> (color->x);
    out->g            = static_cast<uint16_t> (color->y);
    out->b            = static_cast<uint16_t> (color->z);
}
} // namespace subsampling

//------------------------------------------------------------------------------------------------------------------

namespace Kernel {

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
void firstPointSubsamplingNotAveraged (const KernelConfig& config, Arguments&&... args)
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
        subsampling::kernelFirstPointSubsampleNotAveraged<float, uint8_t>
                <<<grid, block>>> (std::forward<Arguments> (args)...);
    }
    else
    {
        subsampling::kernelFirstPointSubsampleNotAveraged<double, uint8_t>
                <<<grid, block>>> (std::forward<Arguments> (args)...);
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