#include "pointcloud.h"
#include "../tools.cuh"
#include "../timing.cuh"


__global__ void kernelMerging(
        Chunk *grid,
        uint64_t *globalChunkCounter,
        uint64_t newCellAmount,
        uint64_t newGridSize,
        uint64_t oldGridSize,
        uint64_t threshold,
        uint64_t cellOffsetNew,
        uint64_t cellOffsetOld
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

    Chunk *ptr = grid + cellOffsetOld + (z * oldXY * 2) + (y * oldGridSize * 2) + (x * 2);
    Chunk *chunk_0_0_0 = ptr;
    Chunk *chunk_0_0_1 = (chunk_0_0_0 + 1);
    Chunk *chunk_0_1_0 = (chunk_0_0_0 + oldGridSize);
    Chunk *chunk_0_1_1 = (chunk_0_1_0 + 1);
    Chunk *chunk_1_0_0 = (ptr + oldXY);
    Chunk *chunk_1_0_1 = (chunk_1_0_0 + 1);
    Chunk *chunk_1_1_0 = (chunk_1_0_0 + oldGridSize);
    Chunk *chunk_1_1_1 = (chunk_1_1_0 + 1);

    auto sum =
            chunk_0_0_0->count +
            chunk_0_0_1->count +
            chunk_0_1_0->count +
            chunk_0_1_1->count +
            chunk_1_0_0->count +
            chunk_1_0_1->count +
            chunk_1_1_0->count +
            chunk_1_1_1->count;

    bool containsFinalizedCells = chunk_0_0_0->isFinished ||
                                  chunk_0_0_1->isFinished ||
                                  chunk_0_1_0->isFinished ||
                                  chunk_0_1_1->isFinished ||
                                  chunk_1_0_0->isFinished ||
                                  chunk_1_0_1->isFinished ||
                                  chunk_1_1_0->isFinished ||
                                  chunk_1_1_1->isFinished;

    bool isFinalized = (sum >= threshold) || containsFinalizedCells;


    grid[cellOffsetNew + index].count = !isFinalized ? sum : 0;
    grid[cellOffsetNew + index].isFinished = isFinalized;

    chunk_0_0_0->dst = isFinalized ? nullptr : (grid + cellOffsetNew + index);
    chunk_0_0_1->dst = chunk_0_0_0->dst;
    chunk_0_1_0->dst = chunk_0_0_0->dst;
    chunk_0_1_1->dst = chunk_0_0_0->dst;
    chunk_1_0_0->dst = chunk_0_0_0->dst;
    chunk_1_0_1->dst = chunk_0_0_0->dst;
    chunk_1_1_0->dst = chunk_0_0_0->dst;
    chunk_1_1_1->dst = chunk_0_0_0->dst;

    chunk_0_0_0->isFinished = isFinalized;
    chunk_0_0_1->isFinished = chunk_0_0_0->isFinished;
    chunk_0_1_0->isFinished = chunk_0_0_0->isFinished;
    chunk_0_1_1->isFinished = chunk_0_0_0->isFinished;
    chunk_1_0_0->isFinished = chunk_0_0_0->isFinished;
    chunk_1_0_1->isFinished = chunk_0_0_0->isFinished;
    chunk_1_1_0->isFinished = chunk_0_0_0->isFinished;
    chunk_1_1_1->isFinished = chunk_0_0_0->isFinished;

    if(isFinalized) {
        uint64_t i = atomicAdd(globalChunkCounter, sum);
        chunk_0_0_0->treeIndex = i;
        i += chunk_0_0_0->count;
        chunk_0_0_1->treeIndex = i;
        i += chunk_0_0_1->count;
        chunk_0_1_0->treeIndex = i;
        i += chunk_0_1_0->count;
        chunk_0_1_1->treeIndex = i;
        i += chunk_0_1_1->count;
        chunk_1_0_0->treeIndex = i;
        i += chunk_1_0_0->count;
        chunk_1_0_1->treeIndex = i;
        i += chunk_1_0_1->count;
        chunk_1_1_0->treeIndex = i;
        i += chunk_1_1_0->count;
        chunk_1_1_1->treeIndex = i;
    }
}

void PointCloud::performCellMerging(uint64_t threshold) {
    itsTreeData = make_unique<CudaArray<Vector3>>(itsData->pointCount());
    auto counter = make_unique<CudaArray<uint64_t>>(1);
    cudaMemset (counter->devicePointer(), 0, 1 * sizeof(uint64_t));

    uint64_t cellOffsetNew = 0;
    uint64_t cellOffsetOld = 0;
    for(uint64_t gridSize = itsGridBaseSideLength; gridSize > 1; gridSize >>= 1) {
        auto newCellAmount = static_cast<uint64_t>(pow(gridSize, 3) / 8);

        cellOffsetNew += static_cast<uint64_t >(pow(gridSize, 3));

        dim3 grid, block;
        tools::create1DKernel(block, grid, newCellAmount);

        tools::KernelTimer timer;
        timer.start();
        kernelMerging <<<  grid, block >>> (
                itsGrid->devicePointer(),
                counter->devicePointer(),
                newCellAmount,
                gridSize>>1,
                gridSize,
                threshold,
                cellOffsetNew,
                cellOffsetOld);
        timer.stop();

        cellOffsetOld = cellOffsetNew;
        spdlog::info("'performCellMerging' for a grid size of {} took {:f} [ms]", gridSize, timer.getMilliseconds());
    }
}