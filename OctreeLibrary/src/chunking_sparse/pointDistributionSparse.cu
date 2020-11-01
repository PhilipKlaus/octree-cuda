#include "../pointcloud.h"
#include "../tools.cuh"
#include "../timing.cuh"


__global__ void kernelDistributingSparse (
        Chunk *octreeSparse,
        Vector3 *cloud,
        uint32_t *dataLUT,
        int *denseToSparseLUT,
        uint32_t *tmpIndexRegister,
        PointCloudMetadata metadata,
        uint32_t gridSize
) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= metadata.pointAmount) {
        return;
    }
    Vector3 point = cloud[index];

    auto denseVoxelIndex = tools::calculateGridIndex(point, metadata, gridSize);
    auto sparseVoxelIndex = denseToSparseLUT[denseVoxelIndex];

    bool isFinished = octreeSparse[sparseVoxelIndex].isFinished;

    while(!isFinished) {
        sparseVoxelIndex = octreeSparse[sparseVoxelIndex].parentChunkIndex;
        isFinished = octreeSparse[sparseVoxelIndex].isFinished;
    }

    uint32_t dataIndexWithinChunk = atomicAdd(tmpIndexRegister + sparseVoxelIndex, 1);
    dataLUT[octreeSparse[sparseVoxelIndex].chunkDataIndex + dataIndexWithinChunk] = index;
}

void PointCloud::distributePointsSparse() {

    // Create temporary indexRegister for assigning an index for each point within its chunk area
    auto cellAmountSparse = itsCellAmountSparse->toHost()[0];
    auto tmpIndexRegister = make_unique<CudaArray<uint32_t>>(cellAmountSparse, "tmpIndexRegister");
    gpuErrchk(cudaMemset (tmpIndexRegister->devicePointer(), 0, cellAmountSparse * sizeof(uint32_t)));

    // Calculate kernel dimensions
    dim3 grid, block;
    tools::create1DKernel(block, grid, itsCloudData->pointCount());

    // Call distribution kernel
    tools::KernelTimer timer;
    timer.start();
    kernelDistributingSparse <<<  grid, block >>> (
            itsOctreeSparse->devicePointer(),
            itsCloudData->devicePointer(),
            itsDataLUT->devicePointer(),
            itsDenseToSparseLUT->devicePointer(),
            tmpIndexRegister->devicePointer(),
            itsMetadata,
            itsGridBaseSideLength);
    timer.stop();
    gpuErrchk(cudaGetLastError());

    itsDistributionTime = timer.getMilliseconds();
    spdlog::info("'distributePointsSparse' took {:f} [ms]", itsDistributionTime);
}
