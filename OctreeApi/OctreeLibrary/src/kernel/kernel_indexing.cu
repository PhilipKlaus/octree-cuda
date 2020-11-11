#include <kernel_indexing.cuh>
#include <tools.cuh>
#include <timing.cuh>

// Move point indices from old (child LUT) to new (parent LUT)
__global__ void indexing::kernelDistributeSubsamples(
        Vector3 *cloud,
        uint32_t *childDataLUT,
        uint32_t childDataLUTStart,
        uint32_t *parentDataLUT,
        uint32_t *countingGrid,
        int *denseToSparseLUT,
        PointCloudMetadata metadata,
        uint32_t gridSideLength
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= metadata.pointAmount) {
        return;
    }
    Vector3 point = cloud[childDataLUT[childDataLUTStart + index]];

    // 1. Calculate the index within the dense grid of the subsample
    auto denseVoxelIndex = tools::calculateGridIndex(point, metadata, gridSideLength);

    // 2. We are only interested in the first point within a cell -> reset the countinGrid implicitly
    //auto oldIndex = atomicCAS((countingGrid + denseVoxelIndex), 1, 0);
    auto oldIndex = atomicAdd((countingGrid + denseVoxelIndex), 1);

    // 3. If the thread is the first one ->
    //      3.1 store the child lut table index in the parent lut
    //      3.2 'delete' the point within the child lut by invalidating its index entry
    if(oldIndex == 0) {
        parentDataLUT[denseToSparseLUT[denseVoxelIndex]] = childDataLUT[childDataLUTStart + index];
        childDataLUT[childDataLUTStart + index] = INVALID_INDEX;
    }
}

__global__ void indexing::kernelSimpleSubsampling(
        Vector3 *cloud,
        uint32_t *cloudDataLUT,
        uint32_t dataLUTStartIndex,
        uint32_t *densePointCount,
        int *denseToSparseLUT,
        uint32_t *sparseIndexCounter,
        PointCloudMetadata metadata,
        uint32_t gridSideLength
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= metadata.pointAmount) {
        return;
    }

    Vector3 point = cloud[cloudDataLUT[dataLUTStartIndex + index]];

    // 1. Calculate the index within the dense grid of the subsample
    auto denseVoxelIndex = tools::calculateGridIndex(point, metadata, gridSideLength);

    // 2. We are only interested in the first point within a cell
    //auto oldIndex = atomicCAS((densePointCount + denseVoxelIndex), 0, 1);
    auto oldIndex = atomicAdd((densePointCount + denseVoxelIndex), 1);

    // 3. If the thread is the first one -> increase map from the dense grid to the sparse grid
    if(oldIndex == 0) {
        auto sparseVoxelIndex = atomicAdd(sparseIndexCounter, 1);
        denseToSparseLUT[denseVoxelIndex] = sparseVoxelIndex;
    }
}

float indexing::distributeSubsamples(
        unique_ptr<CudaArray<Vector3>> &cloud,
        unique_ptr<CudaArray<uint32_t>> &childDataLUT,
        uint32_t childDataLUTStart,
        unique_ptr<CudaArray<uint32_t>> &parentDataLUT,
        unique_ptr<CudaArray<uint32_t>> &countingGrid,
        unique_ptr<CudaArray<int>> &denseToSparseLUT,
        PointCloudMetadata metadata,
        uint32_t gridSideLength
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
                    metadata,
                    gridSideLength);
    timer.stop();
    gpuErrchk(cudaGetLastError());

    spdlog::info("'distributeSubsamples' took {:f} [ms]", timer.getMilliseconds());
    return timer.getMilliseconds();
}

float indexing::simpleSubsampling(
        unique_ptr<CudaArray<Vector3>> &cloud,
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
    indexing::kernelSimpleSubsampling << < grid, block >> > (
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

    spdlog::info("'simpleSubsampling' took {:f} [ms]", timer.getMilliseconds());
    return timer.getMilliseconds();
}