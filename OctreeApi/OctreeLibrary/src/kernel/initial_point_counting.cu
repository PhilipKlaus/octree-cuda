#include <chunking.cuh>
#include <tools.cuh>
#include <timing.cuh>


__global__ void chunking::kernelInitialPointCounting(
        Vector3 *cloud,
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
    Vector3 point = cloud[index];

    // 1. Calculate the index within the dense grid
    auto denseVoxelIndex = tools::calculateGridIndex(point, metadata, gridSideLength);

    // 2. Accumulate the counter within the dense cell
    auto oldIndex = atomicAdd((densePointCount + denseVoxelIndex), 1);

    // 3. If the thread is the first one accumulating the counter within the cell -> update the denseToSparseLUT
    if(oldIndex == 0) {
        auto sparseVoxelIndex = atomicAdd(sparseIndexCounter, 1);
        denseToSparseLUT[denseVoxelIndex] = sparseVoxelIndex;
    }
}


float chunking::initialPointCounting(
        unique_ptr<CudaArray<Vector3>> &cloud,
        unique_ptr<CudaArray<uint32_t>> &densePointCount,
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
    chunking::kernelInitialPointCounting <<<  grid, block >>> (
            cloud->devicePointer(),
            densePointCount->devicePointer(),
            denseToSparseLUT->devicePointer(),
            sparseIndexCounter->devicePointer(),
            metadata,
            gridSideLength);
    timer.stop();
    gpuErrchk(cudaGetLastError());

    spdlog::info("'kernelMapCloudToGrid' took {:f} [ms]", timer.getMilliseconds());
    return timer.getMilliseconds();
}
