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


/*
__device__ uint32_t calculateParentDenseIndex(uint32_t indexLower, uint32_t gridSizeUpper, uint32_t offsetUpper) {

    // Calculate the dense x,y,z index within the next hierarchy level
    // ToDo: replace '/2' by shifting or multiplication with 0.5
    auto xy = gridSizeUpper * gridSizeUpper;
    auto z = (indexLower / xy) / 2;
    auto y = ((indexLower - (z * xy)) / gridSizeUpper) / 2;
    auto x = ((indexLower - (z * xy)) % gridSizeUpper) / 2;

    return offsetUpper + (z * xy * 2) + (y * gridSizeUpper * 2) + (x * 2);
}

__global__ void kernelDistributingSparse(
        //-------- Cloud data ------
        Vector3 *cloud,
        uint32_t *dataLUT,

        //-------- Octree -----------
        Chunk *sparseOctree,
        uint32_t *densePointCount,
        int *denseToSparseLUT,

        //------- Metadata ----------
        uint32_t octreeLevels,
        uint32_t threshold,
        PointCloudMetadata metadata,
        uint32_t gridSize
        ) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= metadata.pointAmount) {
        return;
    }

    Vector3 point = cloud[index];

    // 1. Find target chunk for storing the index of the point
    //---------------------------------------------------------
    uint32_t currentLevel = 0; // lower means further down from the top
    uint32_t denseVoxelIndex = 0;
    uint32_t pointCountSum = 0;
    uint32_t cellOffset = 0;
    do {

        // 1.1 Determine current index within the dense grid
        denseVoxelIndex =
                (currentLevel == 0) ?
                tools::calculateGridIndex(point, metadata, gridSize) :
                calculateParentDenseIndex(denseVoxelIndex, gridSize >> 1, cellOffset);

        // 2.1. Fetch point count from densePointCount
        pointCountSum = densePointCount[denseVoxelIndex];

        // 2.2. Update loop parameters;
        cellOffset += static_cast<uint32_t >(pow(static_cast<float>(gridSize), 3.f));
        ++currentLevel;
    }
    while(pointCountSum < threshold && currentLevel < octreeLevels);

    // 3. Fetch sparse destination index
    uint32_t sparseVoxelDestinationIndex = denseToSparseLUT[denseVoxelIndex];

    // 4. Fetch pointer to sparse chunk
    Chunk *chunk = sparseOctree + sparseVoxelDestinationIndex;

    // 5. Update the chunk
    //---------------------

    // 5.1. Update chunk point count
    uint32_t oldValue = atomicCAS(&(chunk->pointCount), 0, densePointCount[denseVoxelIndex]);

    // 5.2. If the thread is the first one in the chunk-> create a
}

void PointCloud::distributePointsSparse() {

    // Calculate kernel dimensions
    dim3 grid, block;
    tools::create1DKernel(block, grid, itsCloudData->pointCount());

    // Distribute points
    tools::KernelTimer timer;
    timer.start();
    kernelDistributingSparse <<<  grid, block >>> (
            itsCloudData->devicePointer(),
            itsDataLUT->devicePointer(),
            itsOctreeSparse->devicePointer(),
            itsDensePointCount->devicePointer(),
            itsDenseToSparseLUT->devicePointer(),
            itsOctreeLevels,
            itsMergingThreshold,
            itsMetadata,
            itsGridBaseSideLength
           );
    timer.stop();
    gpuErrchk(cudaGetLastError());

    itsInitialPointCountTime = timer.getMilliseconds();
    spdlog::info("'distributePointsSparse' took {:f} [ms]", itsInitialPointCountTime);
}
*/