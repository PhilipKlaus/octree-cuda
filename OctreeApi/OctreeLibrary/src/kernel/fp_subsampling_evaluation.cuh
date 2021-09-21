#pragma once
#include "kernel_helpers.cuh"

namespace subsampling {
namespace fp {
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
__global__ void kernelEvaluateSubsamplesAveraged (
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
        int sparseIndex                   = atomicAdd (&(octree[nodeIdx].pointCount), 1);
        denseToSparseLUT[denseVoxelIndex] = sparseIndex;

        // Move subsampled averaging and point-LUT data to parent node
        PointLut* dst = lut + octree[nodeIdx].dataIdx + sparseIndex;
        *dst          = *src;
    }
}


template <typename coordinateType, typename colorType>
__global__ void kernelInterCellAvg (
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


    { // apply averages to neighbourhing cells as well
        double t  = gridding.bbSize / gridding.gridSize;
        double uX = (point->x - gridding.bbMin.x) / t;
        double uY = (point->y - gridding.bbMin.y) / t;
        double uZ = (point->z - gridding.bbMin.z) / t;

        t           = gridding.gridSize - 1.0;
        uint64_t ix = static_cast<int64_t> (fmin (uX, t));
        uint64_t iy = static_cast<int64_t> (fmin (uY, t));
        uint64_t iz = static_cast<int64_t> (fmin (uZ, t));


        // Encode the point color and add it up
        uint64_t encoded = encodeColors (srcBuffer->r, srcBuffer->g, srcBuffer->b);

        bool underflow = false;
        bool overflow  = false;
        for (int64_t ox = -1; ox <= 1; ox++)
        {
            for (int64_t oy = -1; oy <= 1; oy++)
            {
                for (int64_t oz = -1; oz <= 1; oz++)
                {
                    int32_t nx = ix + ox;
                    int32_t ny = iy + oy;
                    int32_t nz = iz + oz;

                    underflow = nx < 0 || ny < 0 || nz < 0;
                    overflow  = nx >= gridding.gridSize || ny >= gridding.gridSize || nz >= gridding.gridSize;

                    uint32_t voxelIndex = static_cast<uint32_t> (
                            nx + ny * gridding.gridSize + nz * gridding.gridSize * gridding.gridSize);

                    if (!underflow && !overflow && averagingGrid[voxelIndex] != 0)
                    {
                        atomicAdd (&(averagingGrid[voxelIndex]), encoded);
                    }
                }
            }
        }
    }
}

} // namespace fp
} // namespace subsampling

//------------------------------------------------------------------------------------------

namespace Kernel {
namespace fp {

template <typename... Arguments>
void evaluateSubsamplesAveraged (const KernelConfig& config, Arguments&&... args)
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
        subsampling::fp::kernelEvaluateSubsamplesAveraged<float, uint8_t>
                <<<grid, block>>> (std::forward<Arguments> (args)...);
    }
    else
    {
        subsampling::fp::kernelEvaluateSubsamplesAveraged<double, uint8_t>
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
void interCellAvg (const KernelConfig& config, Arguments&&... args)
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

    subsampling::fp::kernelInterCellAvg<float, uint8_t><<<grid, block>>> (std::forward<Arguments> (args)...);


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