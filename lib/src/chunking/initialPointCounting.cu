#include "pointcloud.h"
#include "../tools.cuh"
#include "../timing.cuh"


__global__ void kernelCounting(Chunk *grid, Vector3 *cloud, PointCloudMetadata metadata, uint32_t gridSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= metadata.pointAmount) {
        return;
    }
    Vector3 point = cloud[index];

    auto gridIndex = tools::calculateGridIndex(point, metadata, gridSize);

    atomicAdd(&(grid + gridIndex)->pointCount, 1);
}

void PointCloud::initialPointCounting(uint32_t initialDepth) {

    // Precalculate parameters
    itsGridBaseSideLength = static_cast<uint32_t >(pow(2, initialDepth));
    for(uint32_t gridSize = itsGridBaseSideLength; gridSize > 0; gridSize >>= 1) {
        itsCellAmount += static_cast<uint32_t>(pow(gridSize, 3));
    }
    spdlog::info("Overall 'CellAmount' in hierarchical grid {}", itsCellAmount);

    // Create and initialize the complete grid
    itsOctree = make_unique<CudaArray<Chunk>>(itsCellAmount, "grid");
    cudaMemset (itsOctree->devicePointer(), 0, itsCellAmount * sizeof(Chunk));

    // Calculate kernel dimensions
    dim3 grid, block;
    tools::create1DKernel(block, grid, itsCloudData->pointCount());

    // Initial point counting
    tools::KernelTimer timer;
    timer.start();
    kernelCounting <<<  grid, block >>> (
            itsOctree->devicePointer(),
            itsCloudData->devicePointer(),
            itsMetadata,
            itsGridBaseSideLength);
    timer.stop();
    itsInitialPointCountTime = timer.getMilliseconds();
    spdlog::info("'initialPointCounting' took {:f} [ms]", itsInitialPointCountTime);
}