#include "pointcloud.h"
#include "../tools.cuh"
#include "../timing.cuh"


__global__ void kernelDistributing(
        Chunk *grid,
        Vector3 *cloud,
        uint64_t *treeData,
        PointCloudMetadata metadata,
        uint64_t gridSize
        ) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= metadata.pointAmount) {
        return;
    }
    Vector3 point = cloud[index];

    auto gridIndex = tools::calculateGridIndex(point, metadata, gridSize);

    uint64_t dst = gridIndex;
    bool isFinished = grid[dst].isFinished;

    while(!isFinished) {
        dst = grid[dst].parentChunkIndex;
        isFinished = grid[dst].isFinished;
    }

    uint64_t i = atomicAdd(&(grid[dst].indexCount), 1);
    treeData[grid[dst].treeIndex + i] = index;
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
