#include <chunking.cuh>
#include <tools.cuh>
#include <timing.cuh>
#include <cstdint>
#include "../../include/types.h"


__global__ void chunking::kernelInitialPointCounting(
        uint8_t *cloud,
        uint32_t *densePointCount,
        int *denseToSparseLUT,
        uint32_t *sparseIndexCounter,
        PointCloudMetadata metadata,
        uint32_t gridSideLength
) {

    int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
    if(index >= metadata.pointAmount) {
        return;
    }

    Vector3 *point = reinterpret_cast<Vector3 *>(cloud + index * metadata.pointDataStride);

    // 1. Calculate the index within the dense grid
    auto denseVoxelIndex = tools::calculateGridIndex(point, metadata, gridSideLength);

    // 2. Accumulate the counter within the dense cell
    auto oldIndex = atomicAdd((densePointCount + denseVoxelIndex), 1);

    // 3. If the thread is the first one accumulating the counter within the cell -> update the denseToSparseLUT
    if(oldIndex == 0) {
        auto sparseVoxelIndex = atomicAdd(sparseIndexCounter, 1);
        denseToSparseLUT[denseVoxelIndex] = sparseVoxelIndex;
    }
}


float chunking::initialPointCounting(
        unique_ptr<CudaArray<uint8_t>> &cloud,
        unique_ptr<CudaArray<uint32_t>> &densePointCount,
        unique_ptr<CudaArray<int>> &denseToSparseLUT,
        unique_ptr<CudaArray<uint32_t>> &sparseIndexCounter,
        PointCloudMetadata metadata,
        uint32_t gridSideLength
) {
    // Calculate kernel dimensions
    dim3 grid, block;
    tools::create1DKernel(block, grid, metadata.pointAmount);

    // Initial point counting
    tools::KernelTimer timer;
    timer.start();
    chunking::kernelInitialPointCounting <<<  grid, block >>> (
            cloud->devicePointer(),
            densePointCount->devicePointer(),
            denseToSparseLUT->devicePointer(),
            sparseIndexCounter->devicePointer(),
            metadata,
            gridSideLength);
    timer.stop();
    gpuErrchk(cudaGetLastError());
    return timer.getMilliseconds();
}
