#include <chunking.cuh>
#include <tools.cuh>
#include <timing.cuh>

__global__ void chunking::kernelPropagatePointCounts(
        uint32_t *countingGrid,
        int *denseToSparseLUT,
        uint32_t *sparseIndexCounter,
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

    // The new dense index for the actual chunk
    uint32_t denseVoxelIndex = cellOffsetNew + index;

    // Calculate the dense indices of the 8 underlying cells
    uint32_t chunk_0_0_0_index = cellOffsetOld + (coords.z * oldXY * 2) + (coords.y * oldGridSize * 2) + (coords.x * 2);
    uint32_t chunk_0_0_1_index = chunk_0_0_0_index + 1;
    uint32_t chunk_0_1_0_index = chunk_0_0_0_index + oldGridSize;
    uint32_t chunk_0_1_1_index = chunk_0_1_0_index + 1;
    uint32_t chunk_1_0_0_index = chunk_0_0_0_index + oldXY;
    uint32_t chunk_1_0_1_index = chunk_1_0_0_index + 1;
    uint32_t chunk_1_1_0_index = chunk_1_0_0_index + oldGridSize;
    uint32_t chunk_1_1_1_index = chunk_1_1_0_index + 1;

    // Create pointers to the 8 underlying cells
    uint32_t *chunk_0_0_0 = countingGrid + chunk_0_0_0_index;
    uint32_t *chunk_0_0_1 = countingGrid + chunk_0_0_1_index;
    uint32_t *chunk_0_1_0 = countingGrid + chunk_0_1_0_index;
    uint32_t *chunk_0_1_1 = countingGrid + chunk_0_1_1_index;
    uint32_t *chunk_1_0_0 = countingGrid + chunk_1_0_0_index;
    uint32_t *chunk_1_0_1 = countingGrid + chunk_1_0_1_index;
    uint32_t *chunk_1_1_0 = countingGrid + chunk_1_1_0_index;
    uint32_t *chunk_1_1_1 = countingGrid + chunk_1_1_1_index;

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
        countingGrid[denseVoxelIndex] += sum;
        auto sparseVoxelIndex = atomicAdd(sparseIndexCounter, 1);
        denseToSparseLUT[denseVoxelIndex] = sparseVoxelIndex;
    }
}

float chunking::propagatePointCounts(
        unique_ptr<CudaArray<uint32_t>> &countingGrid,
        unique_ptr<CudaArray<int>> &denseToSparseLUT,
        unique_ptr<CudaArray<uint32_t>> &sparseIndexCounter,
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
    chunking::kernelPropagatePointCounts << < grid, block >> > (
            countingGrid->devicePointer(),
            denseToSparseLUT->devicePointer(),
            sparseIndexCounter->devicePointer(),
            newCellAmount,
            newGridSize,
            oldGridSize,
            cellOffsetNew,
            cellOffsetOld);
    timer.stop();
    gpuErrchk(cudaGetLastError());
    spdlog::debug("'propagatePointCounts' for gridSize of {} took {:f} [ms]", newGridSize, timer.getMilliseconds());
    return timer.getMilliseconds();
}