#include "../pointcloud.h"
#include "../tools.cuh"
#include "../timing.cuh"
#include "../defines.cuh"

__global__ void kernelInitializeBaseGridSparse(
        Chunk *octreeSparse,
        uint32_t *densePointCount,
        int *denseToSparseLUT,
        uint32_t cellAmount
        ) {

    int denseVoxelIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(denseVoxelIndex >= cellAmount) {
        return;
    }

    int sparseVoxelIndex = denseToSparseLUT[denseVoxelIndex];

    if(sparseVoxelIndex == -1) {
        return;
    }

    Chunk *chunk = octreeSparse + sparseVoxelIndex;
    chunk->pointCount = densePointCount[denseVoxelIndex];
}

__global__ void kernelInitializeOctreeSparse(
        Chunk *octree,
        uint32_t *densePointCount,
        int *denseToSparseLUT,
        uint32_t *globalChunkCounter,
        uint32_t threshold,
        uint32_t newCellAmount,
        uint32_t newGridSize,
        uint32_t oldGridSize,
        uint32_t cellOffsetNew,
        uint32_t cellOffsetOld
        ) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= newCellAmount) {
        return;
    }

    // 1. Calculate the actual dense index in the octree
    auto xy = newGridSize * newGridSize;
    auto z = index / xy;
    auto y = (index - (z * xy)) / newGridSize;
    auto x = (index - (z * xy)) % newGridSize;

    auto oldXY = oldGridSize * oldGridSize;
    uint32_t denseVoxelIndex = cellOffsetNew + index;

    // 2. Determine the sparse index in the octree
    int sparseVoxelIndex = denseToSparseLUT[denseVoxelIndex];

    // 3. Check if the actual chunk is existing
    if(sparseVoxelIndex == -1) {
        return;
    }

    // 4. If the chunk exists, calculate the dense indices of the 8 underlying cells
    uint32_t chunk_0_0_0_index = cellOffsetOld + (z * oldXY * 2) + (y * oldGridSize * 2) + (x * 2);
    uint32_t chunk_0_0_1_index = chunk_0_0_0_index + 1;
    uint32_t chunk_0_1_0_index = chunk_0_0_0_index + oldGridSize;
    uint32_t chunk_0_1_1_index = chunk_0_1_0_index + 1;
    uint32_t chunk_1_0_0_index = chunk_0_0_0_index + oldXY;
    uint32_t chunk_1_0_1_index = chunk_1_0_0_index + 1;
    uint32_t chunk_1_1_0_index = chunk_1_0_0_index + oldGridSize;
    uint32_t chunk_1_1_1_index = chunk_1_1_0_index + 1;

    // 5. Update the actual (parent) chunk
    Chunk *chunk = octree + sparseVoxelIndex;
    uint32_t pointCount = densePointCount[denseVoxelIndex];
    bool isFinished = (pointCount >= threshold);

    // 5.1. Update the point count
    chunk->pointCount = isFinished? 0 : pointCount;

    // 5.2. Update the isFinished
    chunk->isFinished = isFinished;

    // 5.3. Assign the sparse indices of the children chunks and calculate the amount of children chunks implicitly
    uint32_t childrenChunksCount = 0;
    int sparseChildIndex = (densePointCount[chunk_0_0_0_index] > 0) ? denseToSparseLUT[chunk_0_0_0_index] : -1;
    chunk->childrenChunks[childrenChunksCount] = sparseChildIndex;
    childrenChunksCount += (sparseChildIndex != -1 ? 1 : 0);

    sparseChildIndex = (densePointCount[chunk_0_0_1_index] > 0) ? denseToSparseLUT[chunk_0_0_1_index] : -1;
    chunk->childrenChunks[childrenChunksCount] = sparseChildIndex;
    childrenChunksCount += (sparseChildIndex != -1 ? 1 : 0);

    sparseChildIndex = (densePointCount[chunk_0_1_0_index] > 0) ? denseToSparseLUT[chunk_0_1_0_index] : -1;
    chunk->childrenChunks[childrenChunksCount] = sparseChildIndex;
    childrenChunksCount += (sparseChildIndex != -1 ? 1 : 0);

    sparseChildIndex = (densePointCount[chunk_0_1_1_index] > 0) ? denseToSparseLUT[chunk_0_1_1_index] : -1;
    chunk->childrenChunks[childrenChunksCount] = sparseChildIndex;
    childrenChunksCount += (sparseChildIndex != -1 ? 1 : 0);

    sparseChildIndex = (densePointCount[chunk_1_0_0_index] > 0) ? denseToSparseLUT[chunk_1_0_0_index] : -1;
    chunk->childrenChunks[childrenChunksCount] = sparseChildIndex;
    childrenChunksCount += (sparseChildIndex != -1 ? 1 : 0);

    sparseChildIndex = (densePointCount[chunk_1_0_1_index] > 0) ? denseToSparseLUT[chunk_1_0_1_index] : -1;
    chunk->childrenChunks[childrenChunksCount] = sparseChildIndex;
    childrenChunksCount += (sparseChildIndex != -1 ? 1 : 0);

    sparseChildIndex = (densePointCount[chunk_1_1_0_index] > 0) ? denseToSparseLUT[chunk_1_1_0_index] : -1;
    chunk->childrenChunks[childrenChunksCount] = sparseChildIndex;
    childrenChunksCount += (sparseChildIndex != -1 ? 1 : 0);

    sparseChildIndex = (densePointCount[chunk_1_1_1_index] > 0) ? denseToSparseLUT[chunk_1_1_1_index] : -1;
    chunk->childrenChunks[childrenChunksCount] = sparseChildIndex;
    childrenChunksCount += (sparseChildIndex != -1 ? 1 : 0);

    // 5.4. update the amount of assigned children chunks
    chunk->childrenChunksCount = childrenChunksCount;

    auto sum = 0;
    // 7. Update all children chunks
    for(auto i = 0; i < childrenChunksCount; ++i) {

        // 6.1. Update isFinished in each child
        (octree + chunk->childrenChunks[i])->isFinished = isFinished;

        // 6.3. Assign current sparse chunk index to child as parentChunkIndex
        (octree + chunk->childrenChunks[i])->parentChunkIndex = sparseVoxelIndex;

        sum += (octree + chunk->childrenChunks[i])->pointCount;
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



}

__global__ void kernelEvaluateSparseOctree(
        uint32_t *densePointCount,
        int *denseToSparseLUT,
        uint32_t *sparseIndexCounter,
        uint32_t newCellAmount,
        uint32_t newGridSize,
        uint32_t oldGridSize,
        uint32_t cellOffsetNew,
        uint32_t cellOffsetOld
        ) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= newCellAmount) {
        return;
    }

    auto xy = newGridSize * newGridSize;
    auto z = index / xy;
    auto y = (index - (z * xy)) / newGridSize;
    auto x = (index - (z * xy)) % newGridSize;

    auto oldXY = oldGridSize * oldGridSize;

    // The new dense index for the actual chunk
    uint32_t denseVoxelIndex = cellOffsetNew + index;

    // Calculate the dense indices of the 8 underlying cells
    uint32_t chunk_0_0_0_index = cellOffsetOld + (z * oldXY * 2) + (y * oldGridSize * 2) + (x * 2);
    uint32_t chunk_0_0_1_index = chunk_0_0_0_index + 1;
    uint32_t chunk_0_1_0_index = chunk_0_0_0_index + oldGridSize;
    uint32_t chunk_0_1_1_index = chunk_0_1_0_index + 1;
    uint32_t chunk_1_0_0_index = chunk_0_0_0_index + oldXY;
    uint32_t chunk_1_0_1_index = chunk_1_0_0_index + 1;
    uint32_t chunk_1_1_0_index = chunk_1_0_0_index + oldGridSize;
    uint32_t chunk_1_1_1_index = chunk_1_1_0_index + 1;

    // Create pointers to the 8 underlying cells
    uint32_t *chunk_0_0_0 = densePointCount + chunk_0_0_0_index;
    uint32_t *chunk_0_0_1 = densePointCount + chunk_0_0_1_index;
    uint32_t *chunk_0_1_0 = densePointCount + chunk_0_1_0_index;
    uint32_t *chunk_0_1_1 = densePointCount + chunk_0_1_1_index;
    uint32_t *chunk_1_0_0 = densePointCount + chunk_1_0_0_index;
    uint32_t *chunk_1_0_1 = densePointCount + chunk_1_0_1_index;
    uint32_t *chunk_1_1_0 = densePointCount + chunk_1_1_0_index;
    uint32_t *chunk_1_1_1 = densePointCount + chunk_1_1_1_index;

    // Buffer the point counts within each cell
    uint32_t chunk_0_0_0_count = *chunk_0_0_0;
    uint32_t chunk_0_0_1_count = *chunk_0_0_1;
    uint32_t chunk_0_1_0_count = *chunk_0_1_0;
    uint32_t chunk_0_1_1_count = *chunk_0_1_1;
    uint32_t chunk_1_0_0_count = *chunk_1_0_0;
    uint32_t chunk_1_0_1_count = *chunk_1_0_1;
    uint32_t chunk_1_1_0_count = *chunk_1_1_0;
    uint32_t chunk_1_1_1_count = *chunk_1_1_1;

    // Summarize all children counts
    auto sum =
            chunk_0_0_0_count +
            chunk_0_0_1_count +
            chunk_0_1_0_count +
            chunk_0_1_1_count +
            chunk_1_0_0_count +
            chunk_1_0_1_count +
            chunk_1_1_0_count +
            chunk_1_1_1_count;

    // If sum > 0:
    // 1. Store children count into densePointCount
    // 2. Increment sparseIndexCounter to mark a new cell and to retrieve a dense index
    // 3. Store the new sparse index in the dense->sparse LUT
    if(sum > 0) {
        densePointCount[denseVoxelIndex] += sum;
        auto sparseVoxelIndex = atomicAdd(sparseIndexCounter, 1);
        denseToSparseLUT[denseVoxelIndex] = sparseVoxelIndex;
    }
}

void PointCloud::performCellMergingSparse(uint32_t threshold) {

    itsMergingThreshold = threshold;

    uint32_t cellOffsetNew = 0;
    uint32_t cellOffsetOld = 0;

    // Perform a hierarchicaly merging of the grid cells which results in an octree structure
    for(uint32_t gridSize = itsGridBaseSideLength; gridSize > 1; gridSize >>= 1) {
        auto newCellAmount = static_cast<uint32_t>(pow(gridSize, 3) / 8);

        cellOffsetNew += static_cast<uint32_t >(pow(gridSize, 3));

        dim3 grid, block;
        tools::create1DKernel(block, grid, newCellAmount);

        tools::KernelTimer timer;
        timer.start();
        kernelEvaluateSparseOctree << < grid, block >> > (
                itsDensePointCount->devicePointer(),
                itsDenseToSparseLUT->devicePointer(),
                itsCellAmountSparse->devicePointer(),
                newCellAmount,
                gridSize>>1,
                gridSize,
                cellOffsetNew,
                cellOffsetOld);
        timer.stop();
        gpuErrchk(cudaGetLastError());

        cellOffsetOld = cellOffsetNew;

        itsMergingTime.push_back(timer.getMilliseconds());
        spdlog::info("'EvaluateSparseOctree' for a grid size of {} took {:f} [ms]", gridSize, itsMergingTime.back());
    }

    uint32_t cellAmountSparse = itsCellAmountSparse->toHost()[0];
    itsOctreeSparse = make_unique<CudaArray<Chunk>>(cellAmountSparse, "octreeSparse");

    spdlog::info(
            "Sparse octree cells: {} instead of {} -> Memory saving: {:f} [%] {:f} [GB]",
            cellAmountSparse,
            itsCellAmount,
            (1 - static_cast<float>(cellAmountSparse) / itsCellAmount) * 100,
            static_cast<float>(itsCellAmount - cellAmountSparse) * sizeof(Chunk) / 1000000000.f
    );

    initializeBaseGridSparse();
    initializeOctreeSparse();
}

void PointCloud::initializeBaseGridSparse() {

    auto cellAmount = static_cast<uint32_t >(pow(itsGridBaseSideLength, 3));

    dim3 grid, block;
    tools::create1DKernel(block, grid, cellAmount);

    tools::KernelTimer timer;
    timer.start();
    kernelInitializeBaseGridSparse << < grid, block >> > (
            itsOctreeSparse->devicePointer(),
            itsDensePointCount->devicePointer(),
            itsDenseToSparseLUT->devicePointer(),
            cellAmount);
    timer.stop();
    gpuErrchk(cudaGetLastError());

    auto initBaseGridTime = timer.getMilliseconds(); // ToDo: make member variable
    spdlog::info("'initializeBaseGridSparse' took {:f} [ms]", initBaseGridTime);
}

void PointCloud::initializeOctreeSparse() {

    // Create a temporary counter register for assigning indices for chunks within the 'itsDataLUT' register
    auto globalChunkCounter = make_unique<CudaArray<uint32_t>>(1, "globalChunkCounter");
    gpuErrchk(cudaMemset (globalChunkCounter->devicePointer(), 0, 1 * sizeof(uint32_t)));

    uint32_t cellOffsetNew = 0;
    uint32_t cellOffsetOld = 0;

    // Perform a hierarchicaly merging of the grid cells which results in an octree structure
    for(uint32_t gridSize = itsGridBaseSideLength; gridSize > 1; gridSize >>= 1) {
        auto newCellAmount = static_cast<uint32_t>(pow(gridSize, 3) / 8);

        cellOffsetNew += static_cast<uint32_t >(pow(gridSize, 3));

        dim3 grid, block;
        tools::create1DKernel(block, grid, newCellAmount);

        tools::KernelTimer timer;
        timer.start();
        kernelInitializeOctreeSparse << < grid, block >> > (
                itsOctreeSparse->devicePointer(),
                itsDensePointCount->devicePointer(),
                itsDenseToSparseLUT->devicePointer(),
                globalChunkCounter->devicePointer(),
                itsMergingThreshold,
                newCellAmount,
                gridSize>>1,
                gridSize,
                cellOffsetNew,
                cellOffsetOld);
        timer.stop();
        gpuErrchk(cudaGetLastError());

        cellOffsetOld = cellOffsetNew;

        itsMergingTime.push_back(timer.getMilliseconds());
        spdlog::info("'initializeOctreeSparse' for a grid size of {} took {:f} [ms]", gridSize, itsMergingTime.back());
    }
}
