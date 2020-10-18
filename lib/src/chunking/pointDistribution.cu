#include "pointcloud.h"
#include "../tools.cuh"

__global__ void kernelDistributing(Chunk *grid, Vector3 *cloud, Vector3 *treeData, PointCloudMetadata metadata, uint16_t gridSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= metadata.pointAmount) {
        return;
    }
    Vector3 point = cloud[index];

    auto gridIndex = tools::calculateGridIndex(point, metadata, gridSize);

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
    tools::create1DKernel(block, grid, itsData->pointCount());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernelDistributing <<<  grid, block >>> (
            itsGrid[0]->devicePointer(),
                    itsData->devicePointer(),
                    itsTreeData->devicePointer(),
                    itsMetadata,
                    itsGridSize);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    spdlog::info("'distributePoints' took {:f} [ms]", milliseconds);
}
