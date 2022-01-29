namespace subsampling {
namespace inter {

/**
 * Places a 3-dimensional grid over 8 octree children nodes and perform inter-cell averaging.
 * For this purpose, the color values of a single point are accumulated to the neighbouring
 * averagingGrid cells.

 * @tparam coordinateType The datatype of the 3D coordinates.
 * @tparam colorType The datatype of the point colors.
 * @param outputBuffer The binary output buffer.
 * @param countingGrid A 3-dimensional grid which stores the amount of points per node.
 * @param octree The octree data structure.
 * @param averagingGrid A 3-dimensional grid which stores averaged color information per node.
 * @param lut Stores the point indices of all points within a node.
 * @param cloud The point cloud data.
 * @param gridding Contains gridding related data.
 * @param nodeIdx The actual parent (target) node.
*/
template <typename coordinateType, typename colorType>
__global__ void kernelInterCellAvg (
        OutputBuffer* outputBuffer,
        uint32_t* countingGrid,
        Node* octree,
        uint64_t* averagingGrid,
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
        for (int32_t ox = -1; ox <= 1; ox++)
        {
            for (int32_t oy = -1; oy <= 1; oy++)
            {
                for (int32_t oz = -1; oz <= 1; oz++)
                {
                    int32_t nx = ix + ox;
                    int32_t ny = iy + oy;
                    int32_t nz = iz + oz;

                    underflow = nx < 0 || ny < 0 || nz < 0;
                    overflow  = nx >= gridding.gridSize || ny >= gridding.gridSize || nz >= gridding.gridSize;

                    uint32_t voxelIndex = static_cast<uint32_t> (
                            nx + ny * gridding.gridSize + nz * gridding.gridSize * gridding.gridSize);

                    if (!underflow && !overflow && countingGrid[voxelIndex] != 0)
                    {
                        atomicAdd (&(averagingGrid[voxelIndex]), encoded);
                    }
                }
            }
        }
    }
}

/**
 * Places a 3-dimensional grid over 8 octree children nodes and perform a distance-weighted inter-cell averaging.
 * For this purpose, the color values of a single point are accumulated to the neighbouring
 * averagingGrid cells based on the distance between the point and the cell centers.

 * @tparam coordinateType The datatype of the 3D coordinates.
 * @tparam colorType The datatype of the point colors.
 * @param outputBuffer The binary output buffer.
 * @param countingGrid A 3-dimensional grid which stores the amount of points per node.
 * @param octree The octree data structure.
 * @param averagingGrid A 3-dimensional grid which stores averaged color information per node.
 * @param lut Stores the point indices of all points within a node.
 * @param cloud The point cloud data.
 * @param gridding Contains gridding related data.
 * @param nodeIdx The actual parent (target) node.
*/
template <typename coordinateType, typename colorType>
__global__ void kernelInterCellAvgWeighted (
        OutputBuffer* outputBuffer,
        uint32_t* countingGrid,
        Node* octree,
        float* rgba,
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

    {
        // Calc current cell coordinates (ix, iy, iz)
        double t  = gridding.bbSize / gridding.gridSize;
        double uX = (point->x - gridding.bbMin.x) / t;
        double uY = (point->y - gridding.bbMin.y) / t;
        double uZ = (point->z - gridding.bbMin.z) / t;

        uint64_t ix = static_cast<int64_t> (fmin (uX, gridding.gridSize - 1.0));
        uint64_t iy = static_cast<int64_t> (fmin (uY, gridding.gridSize - 1.0));
        uint64_t iz = static_cast<int64_t> (fmin (uZ, gridding.gridSize - 1.0));

        bool underflow = false;
        bool overflow  = false;
#pragma unroll
        for (int32_t ox = -1; ox <= 1; ox++)
        {
#pragma unroll
            for (int32_t oy = -1; oy <= 1; oy++)
            {
#pragma unroll
                for (int32_t oz = -1; oz <= 1; oz++)
                {
                    int32_t nx = ix + ox;
                    int32_t ny = iy + oy;
                    int32_t nz = iz + oz;

                    underflow = nx < 0 || ny < 0 || nz < 0;
                    overflow  = nx >= gridding.gridSize || ny >= gridding.gridSize || nz >= gridding.gridSize;

                    auto voxelIndex = static_cast<uint32_t> (
                            nx + ny * gridding.gridSize + nz * gridding.gridSize * gridding.gridSize);


                    // Calc distance components to target cell center
                    uX = (gridding.bbMin.x + nx * t + (t * 0.5)) - point->x;
                    uY = (gridding.bbMin.y + ny * t + (t * 0.5)) - point->y;
                    uZ = (gridding.bbMin.z + nz * t + (t * 0.5)) - point->z;

                    // Calc percentage of dist regarding rMax
                    auto res = static_cast<float> (sqrt (uX * uX + uY * uY + uZ * uZ) / gridding.diag_3_3_half);

                    // Calc weight
                    res = exp (-((res * res) * 10.f));

                    // It is important to access countingGrid[voxelIndex] at last because it might be out-of-bounds
                    if (!underflow && !overflow && countingGrid[voxelIndex] != 0)
                    {
                        atomicAdd (&(rgba[voxelIndex * 4]), static_cast<float> (srcBuffer->r) * res);
                        atomicAdd (&(rgba[voxelIndex * 4 + 1]), static_cast<float> (srcBuffer->g) * res);
                        atomicAdd (&(rgba[voxelIndex * 4 + 2]), static_cast<float> (srcBuffer->b) * res);
                        atomicAdd (&(rgba[voxelIndex * 4 + 3]), res);
                    }
                }
            }
        }
    }
}

} // namespace inter

} // namespace subsampling


namespace Kernel {
namespace inter {
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
    if (config.cloudType == CLOUD_FLOAT_UINT8_T)
    {
        subsampling::inter::kernelInterCellAvg<float, uint8_t><<<grid, block>>> (std::forward<Arguments> (args)...);
    }
    else
    {
        subsampling::inter::kernelInterCellAvg<double, uint8_t><<<grid, block>>> (std::forward<Arguments> (args)...);
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
void interCellAvgWeighted (const KernelConfig& config, Arguments&&... args)
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
        subsampling::inter::kernelInterCellAvgWeighted<float, uint8_t>
                <<<grid, block>>> (std::forward<Arguments> (args)...);
    }
    else
    {
        subsampling::inter::kernelInterCellAvgWeighted<double, uint8_t>
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

} // namespace inter
} // namespace Kernel