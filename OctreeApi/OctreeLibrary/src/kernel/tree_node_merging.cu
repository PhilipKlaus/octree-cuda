#include <chunking.cuh>
#include <tools.cuh>
#include <timing.cuh>

// ToDo: Rename file
__global__ void chunking::kernelMergeHierarchical(
        Chunk *octree,
        uint32_t *countingGrid,
        int *denseToSparseLUT,
        int *sparseToDenseLUT,
        uint32_t *globalChunkCounter,
        uint32_t threshold,
        uint32_t newCellAmount,
        uint32_t newGridSize,
        uint32_t oldGridSize,
        uint32_t cellOffsetNew,
        uint32_t cellOffsetOld
) {

    int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

    if(index >= newCellAmount) {
        return;
    }

    // 1. Calculate the actual dense coordinates in the octree
    Vector3i coords{};
    tools::mapFromDenseIdxToDenseCoordinates(coords, index, newGridSize);

    auto oldXY = oldGridSize * oldGridSize;
    uint32_t denseVoxelIndex = cellOffsetNew + index;

    // 2. Determine the sparse index in the octree
    int sparseVoxelIndex = denseToSparseLUT[denseVoxelIndex];

    // 3. Check if the actual chunk is existing
    if(sparseVoxelIndex == -1) {
        return;
    }

    // 4. If the chunk exists, calculate the dense indices of the 8 underlying cells
    uint32_t chunk_0_0_0_index = cellOffsetOld + (coords.z * oldXY * 2) + (coords.y * oldGridSize * 2) + (coords.x * 2);
    uint32_t chunk_0_0_1_index = chunk_0_0_0_index + 1;
    uint32_t chunk_0_1_0_index = chunk_0_0_0_index + oldGridSize;
    uint32_t chunk_0_1_1_index = chunk_0_1_0_index + 1;
    uint32_t chunk_1_0_0_index = chunk_0_0_0_index + oldXY;
    uint32_t chunk_1_0_1_index = chunk_1_0_0_index + 1;
    uint32_t chunk_1_1_0_index = chunk_1_0_0_index + oldGridSize;
    uint32_t chunk_1_1_1_index = chunk_1_1_0_index + 1;

    // 5. Update the actual (parent) chunk
    Chunk *chunk = octree + sparseVoxelIndex;
    uint32_t pointCount = countingGrid[denseVoxelIndex];
    bool isFinished = (pointCount >= threshold);

    // 5.1. Update the point count
    chunk->pointCount = isFinished? 0 : pointCount;
    chunk->isParent = isFinished;

    // 5.2. Update the isFinished
    chunk->isFinished = isFinished;

    // 5.3. Assign the sparse indices of the children chunks and calculate the amount of children chunks implicitly
    uint32_t childrenChunksCount = 0;
    int sparseChildIndex = (countingGrid[chunk_0_0_0_index] > 0) ? denseToSparseLUT[chunk_0_0_0_index] : -1;
    chunk->childrenChunks[childrenChunksCount] = sparseChildIndex;
    childrenChunksCount += (sparseChildIndex != -1 ? 1 : 0);

    sparseChildIndex = (countingGrid[chunk_0_0_1_index] > 0) ? denseToSparseLUT[chunk_0_0_1_index] : -1;
    chunk->childrenChunks[childrenChunksCount] = sparseChildIndex;
    childrenChunksCount += (sparseChildIndex != -1 ? 1 : 0);

    sparseChildIndex = (countingGrid[chunk_0_1_0_index] > 0) ? denseToSparseLUT[chunk_0_1_0_index] : -1;
    chunk->childrenChunks[childrenChunksCount] = sparseChildIndex;
    childrenChunksCount += (sparseChildIndex != -1 ? 1 : 0);

    sparseChildIndex = (countingGrid[chunk_0_1_1_index] > 0) ? denseToSparseLUT[chunk_0_1_1_index] : -1;
    chunk->childrenChunks[childrenChunksCount] = sparseChildIndex;
    childrenChunksCount += (sparseChildIndex != -1 ? 1 : 0);

    sparseChildIndex = (countingGrid[chunk_1_0_0_index] > 0) ? denseToSparseLUT[chunk_1_0_0_index] : -1;
    chunk->childrenChunks[childrenChunksCount] = sparseChildIndex;
    childrenChunksCount += (sparseChildIndex != -1 ? 1 : 0);

    sparseChildIndex = (countingGrid[chunk_1_0_1_index] > 0) ? denseToSparseLUT[chunk_1_0_1_index] : -1;
    chunk->childrenChunks[childrenChunksCount] = sparseChildIndex;
    childrenChunksCount += (sparseChildIndex != -1 ? 1 : 0);

    sparseChildIndex = (countingGrid[chunk_1_1_0_index] > 0) ? denseToSparseLUT[chunk_1_1_0_index] : -1;
    chunk->childrenChunks[childrenChunksCount] = sparseChildIndex;
    childrenChunksCount += (sparseChildIndex != -1 ? 1 : 0);

    sparseChildIndex = (countingGrid[chunk_1_1_1_index] > 0) ? denseToSparseLUT[chunk_1_1_1_index] : -1;
    chunk->childrenChunks[childrenChunksCount] = sparseChildIndex;
    childrenChunksCount += (sparseChildIndex != -1 ? 1 : 0);

    // 5.4. update the amount of assigned children chunks
    chunk->childrenChunksCount = childrenChunksCount;

    // 6. Update all children chunks
    auto sum = 0;
    for(auto i = 0; i < childrenChunksCount; ++i) {

        // 6.1. Update isFinished in each child
        (octree + chunk->childrenChunks[i])->isFinished = isFinished;

        // 6.3. Assign current sparse chunk index to child as parentChunkIndex
        (octree + chunk->childrenChunks[i])->parentChunkIndex = sparseVoxelIndex;

        sum += (octree + chunk->childrenChunks[i])->pointCount;

        assert((octree + chunk->childrenChunks[i])->pointCount != 0);
    }

    // ##################################################################################


    if(isFinished && sum > 0) {
        // 6. Determine the start index inside the dataLUT for all children chunks
        uint32_t dataLUTIndex = atomicAdd(globalChunkCounter, sum);

        for(auto i = 0; i < childrenChunksCount; ++i) {

            // 6.2. Update the exact index for the child within the dataLUT
            (octree + chunk->childrenChunks[i])->chunkDataIndex = dataLUTIndex;
            dataLUTIndex += (octree + chunk->childrenChunks[i])->pointCount;
        }
    }

    // Update sparseToDense LUT
    sparseToDenseLUT[sparseVoxelIndex] = denseVoxelIndex;
}

float chunking::mergeHierarchical(
        unique_ptr<CudaArray<Chunk>> &octree,
        unique_ptr<CudaArray<uint32_t>> &countingGrid,
        unique_ptr<CudaArray<int>> &denseToSparseLUT,
        unique_ptr<CudaArray<int>> &sparseToDenseLUT,
        unique_ptr<CudaArray<uint32_t>> &globalChunkCounter,
        uint32_t threshold,
        uint32_t newCellAmount,
        uint32_t newGridSize,
        uint32_t oldGridSize,
        uint32_t cellOffsetNew,
        uint32_t cellOffsetOld
) {
    dim3 grid, block;
    tools::create1DKernel(block, grid, newCellAmount);

    tools::KernelTimer timer;
    timer.start();
    chunking::kernelMergeHierarchical << < grid, block >> > (
            octree->devicePointer(),
            countingGrid->devicePointer(),
            denseToSparseLUT->devicePointer(),
            sparseToDenseLUT->devicePointer(),
            globalChunkCounter->devicePointer(),
            threshold,
            newCellAmount,
            newGridSize,
            oldGridSize,
            cellOffsetNew,
            cellOffsetOld);
    timer.stop();
    gpuErrchk(cudaGetLastError());
    spdlog::debug("'kernelMergeHierarchical' for gridSize of {} took {:f} [ms]", newGridSize, timer.getMilliseconds());
    return timer.getMilliseconds();
}