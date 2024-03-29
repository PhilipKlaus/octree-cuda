#pragma once

#include "kernel_executor.cuh"
#include "kernel_helpers.cuh"


namespace subsampling {

namespace fp {


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
__global__ void kernelSubsampleAveraged (
        OutputBuffer* outputBuffer,
        uint32_t* countingGrid,
        uint64_t* averagingGrid,
        KernelStructs::Cloud cloud,
        KernelStructs::Gridding gridding,
        BoundingBox cubic,
        PointLut* lut,
        Node* octree,
        uint32_t nodeIdx,
        int lastNode,
        const uint32_t* leafOffset)
{
    unsigned int localPointIdx = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

    int childIdx = octree[nodeIdx].childNodes[blockIdx.z];
    auto* child  = octree + childIdx;

    if (childIdx == -1 || localPointIdx >= child->pointCount)
    {
        return;
    }

    /******************
     * Get Point data *
     * ****************/
    // Get pointer to the output data entry
    PointLut* src = lut + child->dataIdx + localPointIdx;

    // Get the point within the point cloud
    uint8_t* srcPoint = cloud.raw + (*src) * cloud.dataStride;
    auto* point       = reinterpret_cast<Vector3<coordinateType>*> (srcPoint);

    // Calculate the dense and sparse cell index
    auto denseVoxelIndex = mapPointToGrid<coordinateType> (point, gridding);

    /**************************
     * Ceck for 'first point' *
     * ************************/

    // Check if we are the first thread (first point) resetting the counting cell
    uint32_t old = atomicExch ((countingGrid + denseVoxelIndex), 0);

    // If counting cell is already zero -> we are not the first point
    if (old == 0)
    {
        return;
    }

    /**************************
     *  Store point metadata  *
     * ************************/

    int sparseIndex         = atomicAdd (&(octree[nodeIdx].pointCount), 1);
    octree[nodeIdx].dataIdx = calculateWritingPosition (octree, nodeIdx, lastNode, leafOffset);

    // Store the subsampled point in the parent node
    PointLut* dst = lut + octree[nodeIdx].dataIdx + sparseIndex;

    *dst = *src;

    /**********************
     *  Store point data  *
     * ********************/
    uint64_t encoded = averagingGrid[denseVoxelIndex];
    auto amount      = static_cast<uint16_t> (encoded & 0x3FF);

    uint16_t decoded[3] = {
            static_cast<uint16_t> (((encoded >> 46) & 0xFFFF) / amount),
            static_cast<uint16_t> (((encoded >> 28) & 0xFFFF) / amount),
            static_cast<uint16_t> (((encoded >> 10) & 0xFFFF) / amount)};

    // Write coordinates and colors to output buffer
    OutputBuffer* out = outputBuffer + octree[nodeIdx].dataIdx + sparseIndex;
    out->x            = static_cast<int32_t> (floor ((point->x - cubic.min.x) * cloud.scaleFactor.x));
    out->y            = static_cast<int32_t> (floor ((point->y - cubic.min.y) * cloud.scaleFactor.y));
    out->z            = static_cast<int32_t> (floor ((point->z - cubic.min.z) * cloud.scaleFactor.z));
    out->r            = decoded[0];
    out->g            = decoded[1];
    out->b            = decoded[2];

    /********************************
     *  Reset temp. datastructures  *
     * ******************************/
    averagingGrid[denseVoxelIndex] = 0;
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
__global__ void kernelSubsampleAveragedWeighted (
        OutputBuffer* outputBuffer,
        uint32_t* countingGrid,
        float* rgba,
        KernelStructs::Cloud cloud,
        KernelStructs::Gridding gridding,
        BoundingBox cubic,
        PointLut* lut,
        Node* octree,
        uint32_t nodeIdx,
        int lastNode,
        const uint32_t* leafOffset)
{
    unsigned int localPointIdx = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

    int childIdx = octree[nodeIdx].childNodes[blockIdx.z];
    auto* child  = octree + childIdx;

    if (childIdx == -1 || localPointIdx >= child->pointCount)
    {
        return;
    }

    /******************
     * Get Point data *
     * ****************/
    // Get pointer to the output data entry
    PointLut* src = lut + child->dataIdx + localPointIdx;

    // Get the point within the point cloud
    uint8_t* srcPoint = cloud.raw + (*src) * cloud.dataStride;
    auto* point       = reinterpret_cast<Vector3<coordinateType>*> (srcPoint);

    // Calculate the dense and sparse cell index
    auto denseVoxelIndex = mapPointToGrid<coordinateType> (point, gridding);

    /**************************
     * Ceck for 'first point' *
     * ************************/

    // Check if we are the first thread (first point) resetting the counting cell
    uint32_t old = atomicExch ((countingGrid + denseVoxelIndex), 0);

    // If counting cellwsa already zero -> we are not the first point
    if (old == 0)
    {
        return;
    }

    /**************************
     *  Store point metadata  *
     * ************************/

    int sparseIndex         = atomicAdd (&(octree[nodeIdx].pointCount), 1);
    octree[nodeIdx].dataIdx = calculateWritingPosition (octree, nodeIdx, lastNode, leafOffset);

    // Store the subsampled point in the parent node
    PointLut* dst = lut + octree[nodeIdx].dataIdx + sparseIndex;

    *dst = *src;

    /**********************
     *  Store point data  *
     * ********************/

    // Write coordinates and colors to output buffer
    OutputBuffer* out = outputBuffer + octree[nodeIdx].dataIdx + sparseIndex;
    out->x            = static_cast<int32_t> (floor ((point->x - cubic.min.x) * cloud.scaleFactor.x));
    out->y            = static_cast<int32_t> (floor ((point->y - cubic.min.y) * cloud.scaleFactor.y));
    out->z            = static_cast<int32_t> (floor ((point->z - cubic.min.z) * cloud.scaleFactor.z));

    float* r = rgba + denseVoxelIndex * 4;
    float* g = rgba + denseVoxelIndex * 4 + 1;
    float* b = rgba + denseVoxelIndex * 4 + 2;
    float* w = rgba + denseVoxelIndex * 4 + 3;


    out->r = static_cast<uint16_t> (*r / *w);
    out->g = static_cast<uint16_t> (*g / *w);
    out->b = static_cast<uint16_t> (*b / *w);

    /********************************
     *  Reset temp. datastructures  *
     * ******************************/
    *r = 0.f;
    *g = 0.f;
    *b = 0.f;
    *w = 0.f;
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
__global__ void kernelSubsampleNotAveraged (
        OutputBuffer* outputBuffer,
        uint32_t* countingGrid,
        KernelStructs::Cloud cloud,
        KernelStructs::Gridding gridding,
        BoundingBox cubic,
        PointLut* lut,
        Node* octree,
        uint32_t nodeIdx,
        int lastNode,
        const uint32_t* leafOffset)
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

    // Check if we are the first point -> Mark the cell with the value of 'nodeIdx' if not already done
    auto oldIndex = atomicExch ((countingGrid + denseVoxelIndex), nodeIdx);

    // If the actual thread does not handle the randomly chosen point, exit now.
    if (oldIndex == nodeIdx)
    {
        return;
    }

    int sparseIndex = atomicAdd (&(octree[nodeIdx].pointCount), 1);

    // Update writing position for actual subsampled node
    octree[nodeIdx].dataIdx = calculateWritingPosition (octree, nodeIdx, lastNode, leafOffset);

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
} // namespace fp
} // namespace subsampling

//------------------------------------------------------------------------------------------------------------------

namespace Kernel {

namespace fp {


template <typename... Arguments>
void subsampleAveraged (const KernelConfig& config, Arguments&&... args)
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
        subsampling::fp::kernelSubsampleAveraged<float, uint8_t><<<grid, block>>> (std::forward<Arguments> (args)...);
    }
    else
    {
        subsampling::fp::kernelSubsampleAveraged<double, uint8_t><<<grid, block>>> (std::forward<Arguments> (args)...);
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
void subsampleAveragedWeighted (const KernelConfig& config, Arguments&&... args)
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
        subsampling::fp::kernelSubsampleAveragedWeighted<float, uint8_t>
                <<<grid, block>>> (std::forward<Arguments> (args)...);
    }
    else
    {
        subsampling::fp::kernelSubsampleAveragedWeighted<double, uint8_t>
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

template <typename... Arguments>
void subsampleNotAveraged (const KernelConfig& config, Arguments&&... args)
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
        subsampling::fp::kernelSubsampleNotAveraged<float, uint8_t>
                <<<grid, block>>> (std::forward<Arguments> (args)...);
    }
    else
    {
        subsampling::fp::kernelSubsampleNotAveraged<double, uint8_t>
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
} // namespace fp
} // namespace Kernel