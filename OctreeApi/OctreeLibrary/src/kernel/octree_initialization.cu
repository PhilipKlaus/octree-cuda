#include <chunking.cuh>
#include <tools.cuh>
#include <timing.cuh>

__global__ void chunking::kernelInitLowestOctreeHierarchy(
        Chunk *octreeSparse,
        uint32_t *densePointCount,
        int *denseToSparseLUT,
        int *sparseToDenseLUT,
        uint32_t cellAmount
) {

    int denseVoxelIndex = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

    if(denseVoxelIndex >= cellAmount) {
        return;
    }

    int sparseVoxelIndex = denseToSparseLUT[denseVoxelIndex];

    if(sparseVoxelIndex == -1) {
        return;
    }

    // Update sparseToDense LUT
    sparseToDenseLUT[sparseVoxelIndex] = denseVoxelIndex;

    Chunk *chunk = octreeSparse + sparseVoxelIndex;
    chunk->pointCount = densePointCount[denseVoxelIndex];

    assert(chunk->pointCount != 0);
}

float chunking::initLowestOctreeHierarchy(
        unique_ptr<CudaArray<Chunk>> &octree,
        unique_ptr<CudaArray<uint32_t>> &countingGrid,
        unique_ptr<CudaArray<int>> &denseToSparseLUT,
        unique_ptr<CudaArray<int>> &sparseToDenseLUT,
        uint32_t lowestGridSize
) {

    dim3 grid, block;
    tools::create1DKernel(block, grid, lowestGridSize);

    tools::KernelTimer timer;
    timer.start();
    chunking::kernelInitLowestOctreeHierarchy << < grid, block >> > (
            octree->devicePointer(),
            countingGrid->devicePointer(),
            denseToSparseLUT->devicePointer(),
            sparseToDenseLUT->devicePointer(),
            lowestGridSize);
    timer.stop();
    gpuErrchk(cudaGetLastError());
    return timer.getMilliseconds();
}