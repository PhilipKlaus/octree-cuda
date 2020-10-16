#include "pointcloud.h"
#include "tools.cuh"

__global__ void kernelDistributing(Chunk *grid, Vector3 *cloud, Vector3 *treeData, uint32_t pointCount, Vector3 posOffset, Vector3 size, Vector3 minimum, uint16_t gridSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= pointCount) {
        return;
    }

    Vector3 point = cloud[index];

    // Reference to OctreeConverter
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernelDistributing <<<  grid, block >>> (
            itsGrid[0]->devicePointer(),
                    itsData->devicePointer(),
                    itsTreeData->devicePointer(),
                    itsData->pointCount(),
                    itsMetadata.cloudOffset,
                    itsMetadata.boundingBox.size(),
                    itsMetadata.boundingBox.minimum,
                    itsGridSize);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    spdlog::info("'distributePoints' took {:f} [ms]", milliseconds);
}
