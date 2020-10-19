#include "pointcloud.h"
#include "../tools.cuh"
#include "../timing.cuh"


__global__ void kernelDistributing(Chunk *grid, Vector3 *cloud, Vector3 *treeData, PointCloudMetadata metadata, uint64_t gridSize) {
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

    uint64_t i = atomicAdd(&(dst->indexCount), 1);
    treeData[dst->treeIndex + i] = cloud[index];
}

void PointCloud::distributePoints() {

    dim3 grid, block;
    tools::create1DKernel(block, grid, itsData->pointCount());

    tools::KernelTimer timer;
    timer.start();
    kernelDistributing <<<  grid, block >>> (
            itsGrid->devicePointer(),
            itsData->devicePointer(),
            itsTreeData->devicePointer(),
            itsMetadata,
            itsGridBaseSideLength);
    timer.stop();

    spdlog::info("'distributePoints' took {:f} [ms]", timer.getMilliseconds());
}
