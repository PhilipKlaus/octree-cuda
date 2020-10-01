#include "chunking.cuh"

__global__ void kernelCounting(uint32_t *grid, Point *cloud, uint32_t pointCount, Vector3 posOffset, Vector3 size, Vector3 minimum, uint16_t gridSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= pointCount) {
        return;
    }
    Point point = cloud[index];

    // Copied from OctreeConverter
    double dGridSize = gridSize;
    auto X = int32_t((point.x - posOffset.x) / 1);
    auto Y = int32_t((point.y - posOffset.y) / 1);
    auto Z = int32_t((point.z - posOffset.z) / 1);

    double ux = (double(X) * 1 + posOffset.x - minimum.x) / size.x;
    double uy = (double(Y) * 1 + posOffset.y - minimum.y) / size.y;
    double uz = (double(Z) * 1 + posOffset.z - minimum.z) / size.z;

    uint64_t ix = int64_t( fmin (dGridSize * ux, dGridSize - 1.0));
    uint64_t iy = int64_t( fmin (dGridSize * uy, dGridSize - 1.0));
    uint64_t iz = int64_t( fmin (dGridSize * uz, dGridSize - 1.0));

    uint64_t gridIndex = ix + iy * gridSize + iz * gridSize * gridSize;

    atomicAdd(grid + gridIndex, 1);
}
unique_ptr<CudaArray<uint32_t>> initialPointCounting(unique_ptr<CudaArray<Point>> pointCloud, uint32_t gridSize, Vector3 posOffset, Vector3 size, Vector3 minimum) {
    auto cellAmount = static_cast<uint32_t >(pow(gridSize, 3));

    // Create the counting grid
    auto countingGrid = std::make_unique<CudaArray<uint32_t>>(cellAmount);
    cudaMemset (countingGrid->devicePointer(), 0, cellAmount * sizeof(uint32_t));

    dim3 grid, block;
    createThreadPerPointKernel(block, grid, pointCloud->pointCount());

    kernelCounting <<<  grid, block >>> (
            countingGrid->devicePointer(),
            pointCloud->devicePointer(),
            pointCloud->pointCount(),
            posOffset,
            size,
            minimum,
            gridSize);

    return countingGrid;
}