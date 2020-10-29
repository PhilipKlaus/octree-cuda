#include "pointcloud.h"
#include "../tools.cuh"
#include "../timing.cuh"
#include "../defines.cuh"


__global__ void kernelMergingSparse(
        uint32_t *densePointCount,
        int *denseToSparseLUT,
        uint32_t *sparseIndexCounter,
        uint32_t newCellAmount,
        uint32_t newGridSize,
        uint32_t oldGridSize,
        uint32_t threshold,
        uint32_t cellOffsetNew,
        uint32_t cellOffsetOld
        ) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= newCellAmount) {
        return;
    }

    auto xy = newGridSize * newGridSize;
    auto z = index / xy;
    auto y = (index - (z * xy)) / newGridSize;
    auto x = (index - (z * xy)) % newGridSize;

    auto oldXY = oldGridSize * oldGridSize;

    // The new dense index for the actual chunk
    uint32_t denseVoxelIndex = cellOffsetNew + index;

    // Calculate the dense indices of the 8 underlying cells
    uint32_t chunk_0_0_0_index = cellOffsetOld + (z * oldXY * 2) + (y * oldGridSize * 2) + (x * 2);
    uint32_t chunk_0_0_1_index = chunk_0_0_0_index + 1;
    uint32_t chunk_0_1_0_index = chunk_0_0_0_index + oldGridSize;
    uint32_t chunk_0_1_1_index = chunk_0_1_0_index + 1;
    uint32_t chunk_1_0_0_index = chunk_0_0_0_index + oldXY;
    uint32_t chunk_1_0_1_index = chunk_1_0_0_index + 1;
    uint32_t chunk_1_1_0_index = chunk_1_0_0_index + oldGridSize;
    uint32_t chunk_1_1_1_index = chunk_1_1_0_index + 1;

    // Create pointers to the 8 underlying cells
    uint32_t *chunk_0_0_0 = densePointCount + chunk_0_0_0_index;
    uint32_t *chunk_0_0_1 = densePointCount + chunk_0_0_1_index;
    uint32_t *chunk_0_1_0 = densePointCount + chunk_0_1_0_index;
    uint32_t *chunk_0_1_1 = densePointCount + chunk_0_1_1_index;
    uint32_t *chunk_1_0_0 = densePointCount + chunk_1_0_0_index;
    uint32_t *chunk_1_0_1 = densePointCount + chunk_1_0_1_index;
    uint32_t *chunk_1_1_0 = densePointCount + chunk_1_1_0_index;
    uint32_t *chunk_1_1_1 = densePointCount + chunk_1_1_1_index;

    // Buffer the point counts within each cell
    uint32_t chunk_0_0_0_count = *chunk_0_0_0;
    uint32_t chunk_0_0_1_count = *chunk_0_0_1;
    uint32_t chunk_0_1_0_count = *chunk_0_1_0;
    uint32_t chunk_0_1_1_count = *chunk_0_1_1;
    uint32_t chunk_1_0_0_count = *chunk_1_0_0;
    uint32_t chunk_1_0_1_count = *chunk_1_0_1;
    uint32_t chunk_1_1_0_count = *chunk_1_1_0;
    uint32_t chunk_1_1_1_count = *chunk_1_1_1;

    // Summarize all children counts
    auto sum =
            chunk_0_0_0_count +
            chunk_0_0_1_count +
            chunk_0_1_0_count +
            chunk_0_1_1_count +
            chunk_1_0_0_count +
            chunk_1_0_1_count +
            chunk_1_1_0_count +
            chunk_1_1_1_count;

    // If sum > 0:
    // 1. Store children count into densePointCount
    // 2. Increment sparseIndexCounter to mark a new cell and to retrieve a dense index
    // 3. Store the new sparse index in the dense->sparse LUT
    if(sum > 0) {
        densePointCount[denseVoxelIndex] += sum;
        auto sparseVoxelIndex = atomicAdd(sparseIndexCounter, 1);
        denseToSparseLUT[denseVoxelIndex] = sparseVoxelIndex;
    }
}

void PointCloud::performCellMergingSparse(uint32_t threshold) {


    uint32_t cellOffsetNew = 0;
    uint32_t cellOffsetOld = 0;

    // Perform a hierarchicaly merging of the grid cells which results in an octree structure
    for(uint32_t gridSize = itsGridBaseSideLength; gridSize > 1; gridSize >>= 1) {
        auto newCellAmount = static_cast<uint32_t>(pow(gridSize, 3) / 8);

        cellOffsetNew += static_cast<uint32_t >(pow(gridSize, 3));

        dim3 grid, block;
        tools::create1DKernel(block, grid, newCellAmount);

        tools::KernelTimer timer;
        timer.start();
        kernelMergingSparse <<<  grid, block >>> (
                itsDensePointCount->devicePointer(),
                itsDenseToSparseLUT->devicePointer(),
                itsCellAmountSparse->devicePointer(),
                newCellAmount,
                gridSize>>1,
                gridSize,
                threshold,
                cellOffsetNew,
                cellOffsetOld);
        timer.stop();
        gpuErrchk(cudaGetLastError());

        cellOffsetOld = cellOffsetNew;

        itsMergingTime.push_back(timer.getMilliseconds());
        spdlog::info("'performCellMerging' for a grid size of {} took {:f} [ms]", gridSize, itsMergingTime.back());
    }

    uint32_t cellAmountSparse = itsCellAmountSparse->toHost()[0];
    spdlog::info(
            "Sparse octree cells: {} instead of {} -> Memory saving: {:f} [%] {:f} [GB]",
            cellAmountSparse,
            itsCellAmount,
            (1 - static_cast<float>(cellAmountSparse) / itsCellAmount) * 100,
            static_cast<float>(itsCellAmount - cellAmountSparse) * sizeof(Chunk) / 1000000000.f
    );
}
