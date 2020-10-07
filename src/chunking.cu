#include "pointcloud.h"
#include "tools.cuh"

__global__ void kernelCounting(Chunk *grid, Vector3 *cloud, uint32_t pointCount, Vector3 posOffset, Vector3 size, Vector3 minimum, uint16_t gridSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= pointCount) {
        return;
    }
    Vector3 point = cloud[index];

    // Copied from OctreeConverter
    double dGridSize = gridSize;
    auto X = int32_t((point.x - posOffset.x) / 1);
    auto Y = int32_t((point.y - posOffset.y) / 1);
    auto Z = int32_t((point.z - posOffset.z) / 1);

    double ux = (double(X) * 1 + posOffset.x - minimum.x) / size.x;
    double uy = (double(Y) * 1 + posOffset.y - minimum.y) / size.y;
    double uz = (double(Z) * 1 + posOffset.z - minimum.z) / size.z;

    uint64_t ix = int64_t( fmin (dGridSize * ux, dGridSize - 1.0));
    uint64_t iy = int64_t( fmin (dGridSize * uy, dGridSize - 1.0));
    uint64_t iz = int64_t( fmin (dGridSize * uz, dGridSize - 1.0));

    uint64_t gridIndex = ix + iy * gridSize + iz * gridSize * gridSize;

    atomicAdd(&(grid + gridIndex)->count, 1);
}

void PointCloud::initialPointCounting(uint32_t initialDepth, PointCloudMetadata metadata) {
    itsMetadata = metadata;

    itsInitialDepth = initialDepth;
    itsGridSize = pow(2, initialDepth);
    auto cellAmount = static_cast<uint32_t>(pow(itsGridSize, 3));

    // Create the counting grid
    itsGrid.push_back(make_unique<CudaArray<Chunk>>(cellAmount));
    cudaMemset (itsGrid[0]->devicePointer(), 0, cellAmount * sizeof(uint32_t));

    dim3 grid, block;
    createThreadPerPointKernel(block, grid, itsData->pointCount());

    kernelCounting <<<  grid, block >>> (
                    itsGrid[0]->devicePointer(),
                    itsData->devicePointer(),
                    itsData->pointCount(),
                    metadata.cloudOffset,
                    metadata.boundingBox.size(),
                    metadata.boundingBox.minimum,
                    itsGridSize);
}

__global__ void kernelMerging(Chunk *outputGrid, Chunk *inputGrid, Vector3 *treeData, uint32_t *counter, uint32_t newCellAmount, uint32_t newGridSize, uint32_t oldGridSize) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= newCellAmount) {
        return;
    }
    auto THRESHHOLD = 10000;
    auto xy = newGridSize * newGridSize;
    auto z = index / xy;
    auto y = (index - (z * xy)) / newGridSize;
    auto x = (index - (z * xy)) % newGridSize;

    auto oldXY = oldGridSize * oldGridSize;

    Chunk *ptr = inputGrid + (z * oldXY * 2) + (y * oldGridSize * 2) + (x * 2);
    Chunk *chunk_0_0_0 = ptr;
    Chunk *chunk_0_0_1 = chunk_0_0_0 + 1;
    Chunk *chunk_0_1_0 = chunk_0_0_0 + oldGridSize;
    Chunk *chunk_0_1_1 = chunk_0_1_0 + 1;
    Chunk *chunk_1_0_0 = ptr + oldXY;
    Chunk *chunk_1_0_1 = chunk_1_0_0 + 1;
    Chunk *chunk_1_1_0 = chunk_1_0_0 + oldGridSize;
    Chunk *chunk_1_1_1 = chunk_1_1_0 + 1;

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

    bool isFinalized = sum >= THRESHHOLD || containsFinalizedCells;


    outputGrid[index].count = !isFinalized ? sum : outputGrid[index].count;
    outputGrid[index].isFinished = isFinalized;

    chunk_0_0_0->dst = isFinalized /*&& chunk_0_0_0->count > 0*/ ? chunk_0_0_0->dst : (outputGrid + index);
    chunk_0_0_1->dst = isFinalized /*&& chunk_0_0_0->count > 0*/ ? chunk_0_0_0->dst : (outputGrid + index);
    chunk_0_1_0->dst = isFinalized /*&& chunk_0_0_0->count > 0*/ ? chunk_0_0_0->dst : (outputGrid + index);
    chunk_0_1_1->dst = isFinalized /*&& chunk_0_0_0->count > 0*/? chunk_0_0_0->dst : (outputGrid + index);
    chunk_1_0_0->dst = isFinalized /*&& chunk_0_0_0->count > 0*/ ? chunk_0_0_0->dst : (outputGrid + index);
    chunk_1_0_1->dst = isFinalized /*&& chunk_0_0_0->count > 0*/ ? chunk_0_0_0->dst : (outputGrid + index);
    chunk_1_1_0->dst = isFinalized /*&& chunk_0_0_0->count > 0*/ ? chunk_0_0_0->dst : (outputGrid + index);
    chunk_1_1_1->dst = isFinalized /*&& chunk_0_0_0->count > 0*/ ? chunk_0_0_0->dst : (outputGrid + index);

    chunk_0_0_0->isFinished = isFinalized;
    chunk_0_0_1->isFinished = chunk_0_0_0->isFinished;
    chunk_0_1_0->isFinished = chunk_0_0_0->isFinished;
    chunk_0_1_1->isFinished = chunk_0_0_0->isFinished;
    chunk_1_0_0->isFinished = chunk_0_0_0->isFinished;
    chunk_1_0_1->isFinished = chunk_0_0_0->isFinished;
    chunk_1_1_0->isFinished = chunk_0_0_0->isFinished;
    chunk_1_1_1->isFinished = chunk_0_0_0->isFinished;

    if(sum >= THRESHHOLD) {

        uint32_t i = atomicAdd(counter, sum);

        chunk_0_0_0->points = chunk_0_0_0->count > 0 ? (treeData + i) : nullptr;
        chunk_0_0_0->treeIndex = i;
        i += chunk_0_0_0->count;
        chunk_0_0_1->points = chunk_0_0_1->count > 0 ? (treeData + i) : nullptr;
        chunk_0_0_1->treeIndex = i;
        i += chunk_0_0_1->count;
        chunk_0_1_0->points = chunk_0_1_0->count > 0 ? (treeData + i) : nullptr;
        chunk_0_1_0->treeIndex = i;
        i += chunk_0_1_0->count;
        chunk_0_1_1->points = chunk_0_1_1->count > 0 ? (treeData + i) : nullptr;
        chunk_0_1_1->treeIndex = i;
        i += chunk_0_1_1->count;
        chunk_1_0_0->points = chunk_1_0_0->count > 0 ? (treeData + i) : nullptr;
        chunk_1_0_0->treeIndex = i;
        i += chunk_1_0_0->count;
        chunk_1_0_1->points = chunk_1_0_1->count > 0 ? (treeData + i) : nullptr;
        chunk_1_0_1->treeIndex = i;
        i += chunk_1_0_1->count;
        chunk_1_1_0->points = chunk_1_1_0->count > 0 ? (treeData + i) : nullptr;
        chunk_1_1_0->treeIndex = i;
        i += chunk_1_1_0->count;
        chunk_1_1_1->points = chunk_1_1_1->count > 0 ? (treeData + i) : nullptr;
        chunk_1_1_1->treeIndex = i;
    }
}

void PointCloud::performCellMerging() {

    itsTreeData = make_unique<CudaArray<Vector3>>(itsData->pointCount());
    itsCounter = make_unique<CudaArray<uint32_t>>(1);

    int i = 0;
    for(int gridSize = itsGridSize; gridSize > 1; gridSize >>= 1) {
        auto newCellAmount = static_cast<uint32_t>(pow(gridSize, 3) / 8);
        auto newChunks = make_unique<CudaArray<Chunk>>(newCellAmount);

        dim3 grid, block;
        createThreadPerPointKernel(block, grid, newCellAmount);
        kernelMerging <<<  grid, block >>> (newChunks->devicePointer(), itsGrid[i]->devicePointer(), itsTreeData->devicePointer(), itsCounter->devicePointer(), newCellAmount, gridSize>>1, gridSize);

        itsGrid.push_back(move(newChunks));

        cout << "actualGridsize: " << gridSize << " newCellAmount: " << newCellAmount << endl;
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
    double dGridSize = gridSize;
    auto X = int32_t((point.x - posOffset.x) / 1);
    auto Y = int32_t((point.y - posOffset.y) / 1);
    auto Z = int32_t((point.z - posOffset.z) / 1);

    double ux = (double(X) * 1 + posOffset.x - minimum.x) / size.x;
    double uy = (double(Y) * 1 + posOffset.y - minimum.y) / size.y;
    double uz = (double(Z) * 1 + posOffset.z - minimum.z) / size.z;

    uint64_t ix = int64_t( fmin (dGridSize * ux, dGridSize - 1.0));
    uint64_t iy = int64_t( fmin (dGridSize * uy, dGridSize - 1.0));
    uint64_t iz = int64_t( fmin (dGridSize * uz, dGridSize - 1.0));

    uint64_t gridIndex = ix + iy * gridSize + iz * gridSize * gridSize;

    Chunk *dst = (grid + gridIndex);
    bool isFinished = dst->isFinished;
    while(!isFinished) {
        dst = dst->dst;
        isFinished = dst->isFinished;
    }

    uint32_t i = atomicAdd(&(dst->indexCount), 1);
    dst->points[i] = cloud[index];
}

void PointCloud::distributePoints() {

    dim3 grid, block;
    createThreadPerPointKernel(block, grid, itsData->pointCount());

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

/*
__global__ void kernelBoundingBox(Vector3 *cloud, Vector3* globalMin, Vector3* globalMax, uint32_t pointCount) {

    __shared__ Vector3 localMin[BLOCK_SIZE_MAX];
    __shared__ Vector3 localMax[BLOCK_SIZE_MAX];

    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int localIndex = threadIdx.x;
    int pointsInBlock = pointCount - blockIdx.x * blockDim.x;

    int stride = ceil(pointsInBlock / 2.);

    if(threadIdx.x + stride > pointsInBlock) {
        return;
    }

    Vector3 first = cloud[localIndex];
    Vector3 second = threadIdx.x + stride < pointsInBlock ? cloud[localIndex + stride] : cloud[localIndex];

    localMin[localIndex] = Vector3{
        fmin(first.x, second.x),
        fmin(first.y, second.y),
        fmin(first.z, second.z),
    };

    localMax[localIndex] = Vector3{
            fmax(first.x, second.x),
            fmax(first.y, second.y),
            fmax(first.z, second.z),
    };

    __syncthreads();

    for(uint32_t i = stride >> 1; i > 0; i >>= 1) {
        if(localIndex < i) {
            localMin[localIndex] = Vector3{
                    fmin(localMin[localIndex].x, cloud[localIndex + i].x),
                    fmin(localMin[localIndex].y, cloud[localIndex + i].y),
                    fmin(localMin[localIndex].z, cloud[localIndex + i].z),
            };

            localMax[localIndex] = Vector3{
                    fmax(localMax[localIndex].x, cloud[localIndex + i].x),
                    fmax(localMax[localIndex].y, cloud[localIndex + i].y),
                    fmax(localMax[localIndex].z, cloud[localIndex + i].z),
            };
        }
        __syncthreads();
    }

    if (localIndex == 0) {
        globalMin[blockIdx.x] = localMin[0];
        globalMax[blockIdx.x] = localMax[0];
    }
}

BoundingBox PointCloud::calculateBoundingBox() {

    auto threadAmount = static_cast<uint32_t >(ceil(itsData->pointCount() / 2.));

    dim3 grid, block;
    createThreadPerPointKernel(block, grid, threadAmount);

    auto globalMin = make_unique<CudaArray<Vector3>>(grid.x * grid.y); // 1 min entry per block
    auto globalMax = make_unique<CudaArray<Vector3>>(grid.x * grid.y); // 1 max entry per block

    kernelBoundingBox <<<  grid, block >>> (itsData->devicePointer(), globalMin->devicePointer(), globalMax->devicePointer(), itsData->pointCount());

    auto hostMin = globalMin->toHost();
    auto hostMax = globalMax->toHost();

    cout << grid.x * grid.y << endl;
    for(int i = 0; i < grid.x * grid.y; ++i) {
        //cout << "min: " << hostMin[i].x << " " << hostMin[i].y << " " << hostMin[i].z << endl;
       //cout << "max: " << hostMax[i].x << " " << hostMax[i].y << " " << hostMax[i].z << endl;
    }
    return BoundingBox();
}
*/