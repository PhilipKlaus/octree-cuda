#include <denseOctree.h>
#include "tools.cuh"
#include "timing.cuh"


__global__ void kernelCounting(Chunk *grid, Vector3 *cloud, PointCloudMetadata metadata, uint32_t gridSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= metadata.pointAmount) {
        return;
    }
    Vector3 point = cloud[index];

    auto gridIndex = tools::calculateGridIndex(point, metadata, gridSize);

    atomicAdd(&(grid + gridIndex)->pointCount, 1);
}

void DenseOctree::initialPointCounting(uint32_t initialDepth) {

    // Precalculate parameters
    itsGlobalOctreeDepth = initialDepth;
    itsGlobalOctreeBase = static_cast<uint32_t >(pow(2, initialDepth));
    for(uint32_t gridSize = itsGlobalOctreeBase; gridSize > 0; gridSize >>= 1) {
        itsVoxelAmountDense += static_cast<uint32_t>(pow(gridSize, 3));
    }
    spdlog::info("Overall 'CellAmount' in hierarchical grid {}", itsVoxelAmountDense);

    // Create and initialize the complete grid
    itsOctreeDense = make_unique<CudaArray<Chunk>>(itsVoxelAmountDense, "grid");
    gpuErrchk(cudaMemset (itsOctreeDense->devicePointer(), 0, itsVoxelAmountDense * sizeof(Chunk)));

    // Calculate kernel dimensions
    dim3 grid, block;
    tools::create1DKernel(block, grid, itsCloudData->pointCount());

    // Initial point counting
    tools::KernelTimer timer;
    timer.start();
    kernelCounting <<<  grid, block >>> (
            itsOctreeDense->devicePointer(),
            itsCloudData->devicePointer(),
            itsMetadata,
            itsGlobalOctreeBase);
    timer.stop();
    gpuErrchk(cudaGetLastError());

    itsTimeMeasurement.insert(std::make_pair("initialPointCount", timer.getMilliseconds()));
    spdlog::info("'initialPointCounting' took {:f} [ms]", timer.getMilliseconds());
}


