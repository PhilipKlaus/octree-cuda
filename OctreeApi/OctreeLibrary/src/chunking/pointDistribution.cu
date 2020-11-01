#include "../../include/pointcloud.h"
#include "../tools.cuh"
#include "../timing.cuh"


__global__ void kernelDistributing(
        Chunk *grid,
        Vector3 *cloud,
        uint32_t *dataLUT,
        uint32_t *tmpIndexRegister,
        PointCloudMetadata metadata,
        uint32_t gridSize
        ) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= metadata.pointAmount) {
        return;
    }
    Vector3 point = cloud[index];

    auto gridIndex = tools::calculateGridIndex(point, metadata, gridSize);

    uint32_t dstIndex = gridIndex;
    bool isFinished = grid[dstIndex].isFinished;

    while(!isFinished) {
        dstIndex = grid[dstIndex].parentChunkIndex;
        isFinished = grid[dstIndex].isFinished;
    }

    uint32_t i = atomicAdd(&tmpIndexRegister[dstIndex], 1);
    dataLUT[grid[dstIndex].chunkDataIndex + i] = index;
}

void PointCloud::distributePoints() {

    // Create temporary indexRegister for assigning an index for each point within its chunk area
    auto tmpIndexRegister = make_unique<CudaArray<uint32_t>>(itsCellAmount, "tmpIndexRegister");

    // Calculate kernel dimensions
    dim3 grid, block;
    tools::create1DKernel(block, grid, itsCloudData->pointCount());

    // Call distribution kernel
    tools::KernelTimer timer;
    timer.start();
    kernelDistributing <<<  grid, block >>> (
            itsOctree->devicePointer(),
            itsCloudData->devicePointer(),
            itsDataLUT->devicePointer(),
            tmpIndexRegister->devicePointer(),
            itsMetadata,
            itsGridBaseSideLength);
    timer.stop();
    gpuErrchk(cudaGetLastError());

    // Manually delete the original point cloud data on GPU -> it is not needed anymore
    itsCloudData.reset();

    itsDistributionTime = timer.getMilliseconds();
    spdlog::info("'distributePoints' took {:f} [ms]", itsDistributionTime);
}
