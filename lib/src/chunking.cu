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
        kernelMerging <<<  grid, block >>> (newChunks->devicePointer(), itsGrid[i]->devicePointer(), itsCounter->devicePointer(), newCellAmount, gridSize>>1, gridSize, threshold);

        itsGrid.push_back(move(newChunks));

        ++i;
    }
}

__global__ void kernelDistributing(Chunk *grid, Vector3 *cloud, Vector3 *treeData, uint32_t pointCount, Vector3 posOffset, Vector3 size, Vector3 minimum, uint16_t gridSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= pointCount) {
        return;
    }

    Vector3 point = cloud[index];

    // Copied from OctreeConverter
    float dGridSize = gridSize;
    auto X = int32_t((point.x - posOffset.x) / 1);
    auto Y = int32_t((point.y - posOffset.y) / 1);
    auto Z = int32_t((point.z - posOffset.z) / 1);

    float ux = (float(X) * 1 + posOffset.x - minimum.x) / size.x;
    float uy = (float(Y) * 1 + posOffset.y - minimum.y) / size.y;
    float uz = (float(Z) * 1 + posOffset.z - minimum.z) / size.z;

    uint64_t ix = int64_t( fmin (dGridSize * ux, dGridSize - 1.0f));
    uint64_t iy = int64_t( fmin (dGridSize * uy, dGridSize - 1.0f));
    uint64_t iz = int64_t( fmin (dGridSize * uz, dGridSize - 1.0f));

    uint64_t gridIndex = ix + iy * gridSize + iz * gridSize * gridSize;

    Chunk *dst = (grid + gridIndex);
    bool isFinished = dst->isFinished;

    while(!isFinished) {
        dst = dst->dst;
        isFinished = dst->isFinished;
    }

    uint32_t i = atomicAdd(&(dst->indexCount), 1);
    treeData[dst->treeIndex + i] = cloud[index];
}

void PointCloud::distributePoints() {

    dim3 grid, block;
    create1DKernel(block, grid, itsData->pointCount());

    kernelDistributing <<<  grid, block >>> (
            itsGrid[0]->devicePointer(),
            itsData->devicePointer(),
            itsTreeData->devicePointer(),
            itsData->pointCount(),
            itsMetadata.cloudOffset,
            itsMetadata.boundingBox.size(),
            itsMetadata.boundingBox.minimum,
            itsGridSize);
}
