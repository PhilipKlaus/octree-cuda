#include "pointcloud.h"
#include "../tools.cuh"
#include "../timing.cuh"


__global__ void kernelDistributing(
        Chunk *grid,
        Vector3 *cloud,
        Vector3 *treeData,
        uint64_t *tmpIndexRegister,
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

    uint64_t i = atomicAdd(&tmpIndexRegister[dst], 1);
    treeData[grid[dst].chunkDataIndex + i] = cloud[index];
}

void PointCloud::distributePoints() {

    // Create temporary indexRegister for assigning an index for each point within its chunk area
    auto tmpIndexRegister = make_unique<CudaArray<uint64_t>>(itsCellAmount, "tmpIndexRegister");

    // Calculate kernel dimensions
    dim3 grid, block;
    tools::create1DKernel(block, grid, itsCloudData->pointCount());

    // Call distribution kernel
    tools::KernelTimer timer;
    timer.start();
    kernelDistributing <<<  grid, block >>> (
            itsGrid->devicePointer(),
            itsCloudData->devicePointer(),
            itsChunkData->devicePointer(),
            tmpIndexRegister->devicePointer(),
            itsMetadata,
            itsGridBaseSideLength);
    timer.stop();

    // Manually delete the original point cloud data on GPU -> it is not needed anymore
    itsCloudData.reset();

    spdlog::info("'distributePoints' took {:f} [ms]", timer.getMilliseconds());
}
