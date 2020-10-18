#include "pointcloud.h"
#include "../tools.cuh"


__global__ void kernelCounting(Chunk *grid, Vector3 *cloud, PointCloudMetadata metadata, uint16_t gridSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= metadata.pointAmount) {
        return;
    }
    Vector3 point = cloud[index];

    auto gridIndex = tools::calculateGridIndex(point, metadata, gridSize);

    atomicAdd(&(grid + gridIndex)->count, 1);
}

void PointCloud::initialPointCounting(uint32_t initialDepth) {

    itsInitialDepth = initialDepth;
    itsGridSize = pow(2, initialDepth);
    auto cellAmount = static_cast<uint32_t>(pow(itsGridSize, 3));

    // Create the counting grid
    itsGrid.push_back(make_unique<CudaArray<Chunk>>(cellAmount));
    cudaMemset (itsGrid[0]->devicePointer(), 0, cellAmount * sizeof(uint32_t));

    dim3 grid, block;
    tools::create1DKernel(block, grid, itsData->pointCount());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernelCounting <<<  grid, block >>> (
            itsGrid[0]->devicePointer(),
                    itsData->devicePointer(),
                    itsMetadata,
                    itsGridSize);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    spdlog::info("'initialPointCounting' took {:f} [ms]", milliseconds);
}