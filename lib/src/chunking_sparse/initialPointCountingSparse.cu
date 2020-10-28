#include "pointcloud.h"
#include "../tools.cuh"
#include "../timing.cuh"


__global__ void kernelCountingSparse(
        Vector3 *cloud,
        uint32_t *densePointCount,
        int *itsDenseToSparseLUT,
        uint32_t *sparseIndexCounter,
        PointCloudMetadata metadata,
        uint32_t gridSize
        ) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= metadata.pointAmount) {
        return;
    }
    Vector3 point = cloud[index];

    // 1. Calculate the index within the dense grid
    auto denseVoxelIndex = tools::calculateGridIndex(point, metadata, gridSize);

    // 2. Accumulate the counter within the dense cell
    auto oldIndex = atomicAdd((densePointCount + denseVoxelIndex), 1);

    // 3. If the thread is the first one accumulating the counter within the cell -> update the denseToSparseLUT
    if(oldIndex == 0) {
        auto sparseVoxelIndex = atomicAdd(sparseIndexCounter, 1);
        itsDenseToSparseLUT[denseVoxelIndex] = sparseVoxelIndex;
    }
}

void PointCloud::initialPointCountingSparse(uint32_t initialDepth) {

    // Precalculate parameters
    itsGridBaseSideLength = static_cast<uint32_t >(pow(2, initialDepth));
    for(uint32_t gridSize = itsGridBaseSideLength; gridSize > 0; gridSize >>= 1) {
        itsCellAmount += static_cast<uint32_t>(pow(gridSize, 3));
    }
    spdlog::info("Overall 'CellAmount' in hierarchical grid {}", itsCellAmount);

    // Allocate the dense point count
    itsDensePointCount = make_unique<CudaArray<uint32_t>>(itsCellAmount, "reorderedPointAmount");
    gpuErrchk(cudaMemset (itsDensePointCount->devicePointer(), 0, itsCellAmount * sizeof(uint32_t)));

    // Allocate the conversion LUT from dense to sparse
    itsDenseToSparseLUT = make_unique<CudaArray<int>>(itsCellAmount, "conversionLUT");
    gpuErrchk(cudaMemset (itsDenseToSparseLUT->devicePointer(), -1, itsCellAmount * sizeof(int)));

    // Allocate the global sparseIndexCounter
    auto sparseIndexCounter = make_unique<CudaArray<uint32_t>>(1, "sparseIndexCounter");
    gpuErrchk(cudaMemset (sparseIndexCounter->devicePointer(), 0, 1 * sizeof(uint32_t)));

    // Calculate kernel dimensions
    dim3 grid, block;
    tools::create1DKernel(block, grid, itsCloudData->pointCount());

    // Initial point counting
    tools::KernelTimer timer;
    timer.start();
    kernelCountingSparse <<<  grid, block >>> (
                    itsCloudData->devicePointer(),
                    itsDensePointCount->devicePointer(),
                    itsDenseToSparseLUT->devicePointer(),
                    sparseIndexCounter->devicePointer(),
                    itsMetadata,
                    itsGridBaseSideLength);
    gpuErrchk(cudaGetLastError());
    timer.stop();
    itsInitialPointCountTime = timer.getMilliseconds();

    // Increase amount of needed cells for sparse octree
    itsCellAmountSparse += sparseIndexCounter->toHost()[0];

    spdlog::info("'initialPointCounting' took {:f} [ms]", itsInitialPointCountTime);
    spdlog::info(
            "Base grid: {} instead of {} -> Memory saving: {:f} [%] {:f} [Bytes]",
            itsCellAmountSparse,
            pow(itsGridBaseSideLength, 3),
            (1 - static_cast<float>(itsCellAmountSparse) / pow(itsGridBaseSideLength, 3)) * 100,
            static_cast<float>(itsCellAmount - itsCellAmountSparse) * sizeof(Chunk) / 1000000000.f
            );
}