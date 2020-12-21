#include "pseudo_random_subsampling.cuh"
#include "../../include/types.h"
#include <tools.cuh>
#include <timing.cuh>


// Move point indices from old (child LUT) to new (parent LUT)
__global__ void pseudo__random_subsampling::kernelDistributeSubsamples(
        uint8_t *cloud,
        uint32_t *childDataLUT,
        uint32_t childDataLUTStart,
        uint32_t *parentDataLUT,
        uint32_t *countingGrid,
        int *denseToSparseLUT,
        uint32_t *sparseIndexCounter,
        PointCloudMetadata metadata,
        uint32_t gridSideLength,
        uint32_t *randomIndices
) {
    int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
    if(index >= metadata.pointAmount) {
        return;
    }

    auto *point = reinterpret_cast<Vector3 *>(cloud + childDataLUT[childDataLUTStart + index] * metadata.pointDataStride);


    // 1. Calculate the index within the dense grid of the subsample
    auto denseVoxelIndex = tools::calculateGridIndex(point, metadata, gridSideLength);

    // 2. We are only interested in the last point within a node -> Implicitly reset the countingGrid
    auto oldIndex = atomicSub((countingGrid + denseVoxelIndex), 1);

    // 3. If the thread is the first one ->
    //      3.1 store the child lut table index in the parent lut
    //      3.2 'delete' the point within the child lut by invalidating its index entry

    int sparseIndex = denseToSparseLUT[denseVoxelIndex];

    if(oldIndex == randomIndices[sparseIndex]) {
        parentDataLUT[sparseIndex] = childDataLUT[childDataLUTStart + index];
        childDataLUT[childDataLUTStart + index] = INVALID_INDEX;
    }
    if(oldIndex == 1) {
        // Reset data structures
        denseToSparseLUT[denseVoxelIndex] = -1;
        *sparseIndexCounter = 0;
    }
}

// http://ianfinlayson.net/class/cpsc425/notes/cuda-random
__global__ void pseudo__random_subsampling::kernelInitRandoms(
        unsigned int seed,
        curandState_t *states,
        uint32_t nodeAmount) {

    int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

    if(index >= nodeAmount) {
        return;
    }

    /* we have to initialize the state */
    curand_init(
            seed,
            index,
            0,
            &states[index]);
}

__global__ void pseudo__random_subsampling::kernelGenerateRandoms(
        curandState_t* states,
        uint32_t *randomIndices,
        const int *denseToSparseLUT,
        const uint32_t *countingGrid,
        uint32_t gridNodes) {

    int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

    if(index >= gridNodes) {
        return;
    }

    int sparseIndex = denseToSparseLUT[index];

    if(sparseIndex > -1) {
        randomIndices[sparseIndex] = static_cast<uint32_t>(ceil(curand_uniform(&states[sparseIndex]) * countingGrid[index]));
    }
}

__global__ void pseudo__random_subsampling::kernelSubsample(
        uint8_t *cloud,
        uint32_t *cloudDataLUT,
        uint32_t dataLUTStartIndex,
        uint32_t *densePointCount,
        int *denseToSparseLUT,
        uint32_t *sparseIndexCounter,
        PointCloudMetadata metadata,
        uint32_t gridSideLength
) {
    int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
    if(index >= metadata.pointAmount) {
        return;
    }

    //Vector3 point = cloud[cloudDataLUT[dataLUTStartIndex + index]];
    auto *point = reinterpret_cast<Vector3 *>(cloud + cloudDataLUT[dataLUTStartIndex + index] * metadata.pointDataStride);

    // 1. Calculate the index within the dense grid of the subsample
    auto denseVoxelIndex = tools::calculateGridIndex(point, metadata, gridSideLength);

    // 2. We are only interested in the first point within a cell
    auto oldIndex = atomicAdd((densePointCount + denseVoxelIndex), 1);

    // 3. If the thread is the first one -> increase map from the dense grid to the sparse grid
    if(oldIndex == 0) {
        auto sparseVoxelIndex = atomicAdd(sparseIndexCounter, 1);
        denseToSparseLUT[denseVoxelIndex] = sparseVoxelIndex;
    }
}

float pseudo__random_subsampling::distributeSubsamples(
        unique_ptr<CudaArray<uint8_t>> &cloud,
        unique_ptr<CudaArray<uint32_t>> &childDataLUT,
        uint32_t childDataLUTStart,
        unique_ptr<CudaArray<uint32_t>> &parentDataLUT,
        unique_ptr<CudaArray<uint32_t>> &countingGrid,
        unique_ptr<CudaArray<int>> &denseToSparseLUT,
        unique_ptr<CudaArray<uint32_t>> &sparseIndexCounter,
        PointCloudMetadata metadata,
        uint32_t gridSideLength,
        unique_ptr<CudaArray<uint32_t>> &randomIndices
) {
    // Calculate kernel dimensions
    dim3 grid, block;
    tools::create1DKernel(block, grid, metadata.pointAmount);

    // Initial point counting
    tools::KernelTimer timer;
    timer.start();
    kernelDistributeSubsamples <<<  grid, block >>> (
            cloud->devicePointer(),
                    childDataLUT->devicePointer(),
                    childDataLUTStart,
                    parentDataLUT->devicePointer(),
                    countingGrid->devicePointer(),
                    denseToSparseLUT->devicePointer(),
                    sparseIndexCounter->devicePointer(),
                    metadata,
                    gridSideLength,
                    randomIndices->devicePointer());
    timer.stop();
    gpuErrchk(cudaGetLastError());

    spdlog::debug("'kernelDistributeSubsamples' took {:f} [ms]", timer.getMilliseconds());
    return timer.getMilliseconds();
}

float pseudo__random_subsampling::generateRandoms(
        const unique_ptr<CudaArray<curandState_t>> &states,
        unique_ptr<CudaArray<uint32_t>> &randomIndices,
        const unique_ptr<CudaArray<int>> &denseToSparseLUT,
        const unique_ptr<CudaArray<uint32_t>> &countingGrid,
        uint32_t gridNodes) {

    // Calculate kernel dimensions
    dim3 grid, block;
    tools::create1DKernel(block, grid, gridNodes);

    // Initial point counting
    tools::KernelTimer timer;
    timer.start();
    kernelGenerateRandoms << < grid, block >> > (states->devicePointer(), randomIndices->devicePointer(), denseToSparseLUT->devicePointer(), countingGrid->devicePointer(), gridNodes);
    timer.stop();
    gpuErrchk(cudaGetLastError());

    spdlog::debug("'kernelGenerateRandoms' took {:f} [ms]", timer.getMilliseconds());
    return timer.getMilliseconds();
}

float pseudo__random_subsampling::initRandoms(unsigned int seed, unique_ptr<CudaArray<curandState_t>> &states, uint32_t nodeAmount) {

    // Calculate kernel dimensions
    dim3 grid, block;
    tools::create1DKernel(block, grid, nodeAmount);

    // Initial point counting
    tools::KernelTimer timer;
    timer.start();
    kernelInitRandoms << < grid, block >> > (seed, states->devicePointer(), nodeAmount);
    timer.stop();
    gpuErrchk(cudaGetLastError());

    spdlog::debug("'kernelInitRandoms' took {:f} [ms]", timer.getMilliseconds());
    return timer.getMilliseconds();
}

float pseudo__random_subsampling::subsample(
        unique_ptr<CudaArray<uint8_t>> &cloud,
        unique_ptr<CudaArray<uint32_t>> &cloudDataLUT,
        uint32_t dataLUTStartIndex,
        unique_ptr<CudaArray<uint32_t>> &countingGrid,
        unique_ptr<CudaArray<int>> &denseToSparseLUT,
        unique_ptr<CudaArray<uint32_t>> &sparseIndexCounter,
        PointCloudMetadata metadata,
        uint32_t gridSideLength
) {
    // Calculate kernel dimensions
    dim3 grid, block;
    tools::create1DKernel(block, grid, metadata.pointAmount);

    // Initial point counting
    tools::KernelTimer timer;
    timer.start();
    kernelSubsample << < grid, block >> > (
            cloud->devicePointer(),
                    cloudDataLUT->devicePointer(),
                    dataLUTStartIndex,
                    countingGrid->devicePointer(),
                    denseToSparseLUT->devicePointer(),
                    sparseIndexCounter->devicePointer(),
                    metadata,
                    gridSideLength);
    timer.stop();
    gpuErrchk(cudaGetLastError());

    spdlog::debug("'kernelSubsample' took {:f} [ms]", timer.getMilliseconds());
    return timer.getMilliseconds();
}