#include <chunking.cuh>
#include <tools.cuh>
#include <timing.cuh>

__global__ void chunking::kernelDistributePoints (
        Chunk *octree,
        uint8_t *cloud,
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
    //Vector3 point = cloud[index];
    Vector3 *point = reinterpret_cast<Vector3 *>(cloud + index * metadata.pointDataStride);


    auto denseVoxelIndex = tools::calculateGridIndex(point, metadata, gridSize);
    auto sparseVoxelIndex = denseToSparseLUT[denseVoxelIndex];

    bool isFinished = octree[sparseVoxelIndex].isFinished;

    while(!isFinished) {
        sparseVoxelIndex = octree[sparseVoxelIndex].parentChunkIndex;
        isFinished = octree[sparseVoxelIndex].isFinished;
    }

    uint32_t dataIndexWithinChunk = atomicAdd(tmpIndexRegister + sparseVoxelIndex, 1);
    dataLUT[octree[sparseVoxelIndex].chunkDataIndex + dataIndexWithinChunk] = index;
}

float chunking::distributePoints(
        unique_ptr<CudaArray<Chunk>> &octree,
        unique_ptr<CudaArray<uint8_t>> &cloud,
        unique_ptr<CudaArray<uint32_t>> &dataLUT,
        unique_ptr<CudaArray<int>> &denseToSparseLUT,
        unique_ptr<CudaArray<uint32_t>> &tmpIndexRegister,
        PointCloudMetadata metadata,
        uint32_t gridSize
) {
    // Calculate kernel dimensions
    dim3 grid, block;
    tools::create1DKernel(block, grid, cloud->pointCount());

    // Call distribution kernel
    tools::KernelTimer timer;
    timer.start();
    chunking::kernelDistributePoints <<<  grid, block >>> (
            octree->devicePointer(),
            cloud->devicePointer(),
            dataLUT->devicePointer(),
            denseToSparseLUT->devicePointer(),
            tmpIndexRegister->devicePointer(),
            metadata,
            gridSize);
    timer.stop();
    gpuErrchk(cudaGetLastError());
    return timer.getMilliseconds();
}