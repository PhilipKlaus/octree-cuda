#include "pointcloud.h"
#include "../tools.cuh"
#include "../timing.cuh"
#include "../defines.cuh"


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

    uint64_t chunk_0_0_0_index = cellOffsetOld + (z * oldXY * 2) + (y * oldGridSize * 2) + (x * 2);
    uint64_t chunk_0_0_1_index = chunk_0_0_0_index + 1;
    uint64_t chunk_0_1_0_index = chunk_0_0_0_index + oldGridSize;
    uint64_t chunk_0_1_1_index = chunk_0_1_0_index + 1;
    uint64_t chunk_1_0_0_index = chunk_0_0_0_index + oldXY;
    uint64_t chunk_1_0_1_index = chunk_1_0_0_index + 1;
    uint64_t chunk_1_1_0_index = chunk_1_0_0_index + oldGridSize;
    uint64_t chunk_1_1_1_index = chunk_1_1_0_index + 1;

    Chunk *chunk_0_0_0 = grid + chunk_0_0_0_index;
    Chunk *chunk_0_0_1 = grid + chunk_0_0_1_index;
    Chunk *chunk_0_1_0 = grid + chunk_0_1_0_index;
    Chunk *chunk_0_1_1 = grid + chunk_0_1_1_index;
    Chunk *chunk_1_0_0 = grid + chunk_1_0_0_index;
    Chunk *chunk_1_0_1 = grid + chunk_1_0_1_index;
    Chunk *chunk_1_1_0 = grid + chunk_1_1_0_index;
    Chunk *chunk_1_1_1 = grid + chunk_1_1_1_index;

    uint64_t chunk_0_0_0_count = chunk_0_0_0->pointCount;
    uint64_t chunk_0_0_1_count = chunk_0_0_1->pointCount;
    uint64_t chunk_0_1_0_count = chunk_0_1_0->pointCount;
    uint64_t chunk_0_1_1_count = chunk_0_1_1->pointCount;
    uint64_t chunk_1_0_0_count = chunk_1_0_0->pointCount;
    uint64_t chunk_1_0_1_count = chunk_1_0_1->pointCount;
    uint64_t chunk_1_1_0_count = chunk_1_1_0->pointCount;
    uint64_t chunk_1_1_1_count = chunk_1_1_1->pointCount;

    auto sum =
            chunk_0_0_0_count +
            chunk_0_0_1_count +
            chunk_0_1_0_count +
            chunk_0_1_1_count +
            chunk_1_0_0_count +
            chunk_1_0_1_count +
            chunk_1_1_0_count +
            chunk_1_1_1_count;

    bool containsFinalizedCells = chunk_0_0_0->isFinished ||
                                  chunk_0_0_1->isFinished ||
                                  chunk_0_1_0->isFinished ||
                                  chunk_0_1_1->isFinished ||
                                  chunk_1_0_0->isFinished ||
                                  chunk_1_0_1->isFinished ||
                                  chunk_1_1_0->isFinished ||
                                  chunk_1_1_1->isFinished;

    bool isFinalized = (sum >= threshold) || containsFinalizedCells;

    // Update new (higher-level) chunk
    uint64_t newIndex = cellOffsetNew + index;
    grid[newIndex].pointCount = !isFinalized ? sum : 0;
    grid[newIndex].isFinished = isFinalized;

    grid[newIndex].childrenChunks[0] = chunk_0_0_0_index;
    grid[newIndex].childrenChunks[1] = chunk_0_0_1_index;
    grid[newIndex].childrenChunks[2] = chunk_0_1_0_index;
    grid[newIndex].childrenChunks[3] = chunk_0_1_1_index;
    grid[newIndex].childrenChunks[4] = chunk_1_0_0_index;
    grid[newIndex].childrenChunks[5] = chunk_1_0_1_index;
    grid[newIndex].childrenChunks[6] = chunk_1_1_0_index;
    grid[newIndex].childrenChunks[7] = chunk_1_1_1_index;

    // Update old (8 lower-level) chunks
    chunk_0_0_0->parentChunkIndex = newIndex;
    chunk_0_0_1->parentChunkIndex = newIndex;
    chunk_0_1_0->parentChunkIndex = newIndex;
    chunk_0_1_1->parentChunkIndex = newIndex;
    chunk_1_0_0->parentChunkIndex = newIndex;
    chunk_1_0_1->parentChunkIndex = newIndex;
    chunk_1_1_0->parentChunkIndex = newIndex;
    chunk_1_1_1->parentChunkIndex = newIndex;

    chunk_0_0_0->isFinished = isFinalized;
    chunk_0_0_1->isFinished = isFinalized;
    chunk_0_1_0->isFinished = isFinalized;
    chunk_0_1_1->isFinished = isFinalized;
    chunk_1_0_0->isFinished = isFinalized;
    chunk_1_0_1->isFinished = isFinalized;
    chunk_1_1_0->isFinished = isFinalized;
    chunk_1_1_1->isFinished = isFinalized;

    if(isFinalized && sum > 0) {
        uint64_t i = atomicAdd(globalChunkCounter, sum);
        chunk_0_0_0->chunkDataIndex = i;
        i += chunk_0_0_0_count;
        chunk_0_0_1->chunkDataIndex = i;
        i += chunk_0_0_1_count;
        chunk_0_1_0->chunkDataIndex = i;
        i += chunk_0_1_0_count;
        chunk_0_1_1->chunkDataIndex = i;
        i += chunk_0_1_1_count;
        chunk_1_0_0->chunkDataIndex = i;
        i += chunk_1_0_0_count;
        chunk_1_0_1->chunkDataIndex = i;
        i += chunk_1_0_1_count;
        chunk_1_1_0->chunkDataIndex = i;
        i += chunk_1_1_0_count;
        chunk_1_1_1->chunkDataIndex = i;
    }
}

void PointCloud::performCellMerging(uint64_t threshold) {

    // Create a temporary counter register for assigning indices for chunks within the 'itsChunkData' register
    auto counter = make_unique<CudaArray<uint64_t>>(1, "globalChunkCounter");
    cudaMemset (counter->devicePointer(), 0, 1 * sizeof(uint64_t));

    uint64_t cellOffsetNew = 0;
    uint64_t cellOffsetOld = 0;

    // Perform a hierarchicaly merging of the grid cells which results in an octree structure
    for(uint64_t gridSize = itsGridBaseSideLength; gridSize > 1; gridSize >>= 1) {
        auto newCellAmount = static_cast<uint64_t>(pow(gridSize, 3) / 8);

        cellOffsetNew += static_cast<uint64_t >(pow(gridSize, 3));

        dim3 grid, block;
        tools::create1DKernel(block, grid, newCellAmount);

        tools::KernelTimer timer;
        timer.start();
        kernelMerging <<<  grid, block >>> (
                itsOctree->devicePointer(),
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