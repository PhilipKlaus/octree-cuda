#include "pointcloud.h"
#include "tools.cuh"

__global__ void kernelMerging(Chunk *outputGrid, Chunk *inputGrid, uint32_t *counter, uint32_t newCellAmount, uint32_t newGridSize, uint32_t oldGridSize, uint32_t threshold) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= newCellAmount) {
        return;
    }

    auto xy = newGridSize * newGridSize;
    auto z = index / xy;
    auto y = (index - (z * xy)) / newGridSize;
    auto x = (index - (z * xy)) % newGridSize;

    auto oldXY = oldGridSize * oldGridSize;

    Chunk *ptr = inputGrid + (z * oldXY * 2) + (y * oldGridSize * 2) + (x * 2);
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


    outputGrid[index].count = !isFinalized ? sum : 0;
    outputGrid[index].isFinished = isFinalized;

    chunk_0_0_0->dst = isFinalized ? nullptr : (outputGrid + index);
    chunk_0_0_1->dst = isFinalized ? chunk_0_0_0->dst : (outputGrid + index);
    chunk_0_1_0->dst = isFinalized ? chunk_0_0_0->dst : (outputGrid + index);
    chunk_0_1_1->dst = isFinalized ? chunk_0_0_0->dst : (outputGrid + index);
    chunk_1_0_0->dst = isFinalized ? chunk_0_0_0->dst : (outputGrid + index);
    chunk_1_0_1->dst = isFinalized ? chunk_0_0_0->dst : (outputGrid + index);
    chunk_1_1_0->dst = isFinalized ? chunk_0_0_0->dst : (outputGrid + index);
    chunk_1_1_1->dst = isFinalized ? chunk_0_0_0->dst : (outputGrid + index);

    chunk_0_0_0->isFinished = isFinalized;
    chunk_0_0_1->isFinished = chunk_0_0_0->isFinished;
    chunk_0_1_0->isFinished = chunk_0_0_0->isFinished;
    chunk_0_1_1->isFinished = chunk_0_0_0->isFinished;
    chunk_1_0_0->isFinished = chunk_0_0_0->isFinished;
    chunk_1_0_1->isFinished = chunk_0_0_0->isFinished;
    chunk_1_1_0->isFinished = chunk_0_0_0->isFinished;
    chunk_1_1_1->isFinished = chunk_0_0_0->isFinished;

    if(isFinalized) {
        uint32_t i = atomicAdd(counter, sum);
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

void PointCloud::performCellMerging(uint32_t threshold) {
    itsThreshold = threshold;
    itsTreeData = make_unique<CudaArray<Vector3>>(itsData->pointCount());
    itsCounter = make_unique<CudaArray<uint32_t>>(1);
    cudaMemset (itsCounter->devicePointer(), 0, 1 * sizeof(uint32_t));

    int i = 0;
    for(int gridSize = itsGridSize; gridSize > 1; gridSize >>= 1) {
        auto newCellAmount = static_cast<uint32_t>(pow(gridSize, 3) / 8);
        auto newChunks = make_unique<CudaArray<Chunk>>(newCellAmount);

        dim3 grid, block;
        create1DKernel(block, grid, newCellAmount);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        kernelMerging <<<  grid, block >>> (newChunks->devicePointer(), itsGrid[i]->devicePointer(), itsCounter->devicePointer(), newCellAmount, gridSize>>1, gridSize, threshold);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        spdlog::info("'performCellMerging' for a grid size of {} took {:f} [ms]", gridSize, milliseconds);

        itsGrid.push_back(move(newChunks));

        ++i;
    }
}