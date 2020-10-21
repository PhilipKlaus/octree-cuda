#include "pointcloud.h"
#include "../tools.cuh"
#include "../timing.cuh"


__global__ void kernelCounting(Chunk *grid, Vector3 *cloud, PointCloudMetadata metadata, uint64_t gridSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= metadata.pointAmount) {
        return;
    }
    Vector3 point = cloud[index];

    auto gridIndex = tools::calculateGridIndex(point, metadata, gridSize);

    atomicAdd(&(grid + gridIndex)->pointCount, 1);
}

void PointCloud::initialPointCounting(uint64_t initialDepth) {

    // Precalculate parameters
    itsGridBaseSideLength = static_cast<uint64_t >(pow(2, initialDepth));
    for(uint64_t gridSize = itsGridBaseSideLength; gridSize > 0; gridSize >>= 1) {
        itsCellAmount += static_cast<uint64_t>(pow(gridSize, 3));
    }
    spdlog::info("Overall 'CellAmount' in hierarchical grid {}", itsCellAmount);

    // Create and initialize the complete grid
    itsGrid = make_unique<CudaArray<Chunk>>(itsCellAmount, "grid");
    cudaMemset (itsGrid->devicePointer(), 0, itsCellAmount * sizeof(Chunk));

    // Calculate kernel dimensions
    dim3 grid, block;
    tools::create1DKernel(block, grid, itsCloudData->pointCount());

    // Initial point counting
    tools::KernelTimer timer;
    timer.start();
    kernelCounting <<<  grid, block >>> (
            itsGrid->devicePointer(),
            itsCloudData->devicePointer(),
            itsMetadata,
            itsGridBaseSideLength);
    timer.stop();

    spdlog::info("'initialPointCounting' took {:f} [ms]", timer.getMilliseconds());
}