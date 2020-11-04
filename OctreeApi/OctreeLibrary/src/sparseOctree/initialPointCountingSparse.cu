#include <sparseOctree.h>
#include "tools.cuh"
#include "timing.cuh"
#include "../../include/sparseOctree.h"


__global__ void kernelCountingSparse(
        Vector3 *cloud,
        uint32_t *densePointCount,
        int *itsDenseToSparseLUT,
        uint32_t *sparseIndexCounter,
        PointCloudMetadata metadata,
        uint32_t gridSize
        ) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= metadata.pointAmount) {
        return;
    }
    Vector3 point = cloud[index];

    // 1. Calculate the index within the dense grid
    auto denseVoxelIndex = tools::calculateGridIndex(point, metadata, gridSize);

    // 2. Accumulate the counter within the dense cell
    auto oldIndex = atomicAdd((densePointCount + denseVoxelIndex), 1);

    // 3. If the thread is the first one accumulating the counter within the cell -> update the denseToSparseLUT
    if(oldIndex == 0) {
        auto sparseVoxelIndex = atomicAdd(sparseIndexCounter, 1);
        itsDenseToSparseLUT[denseVoxelIndex] = sparseVoxelIndex;
    }
}

void SparseOctree::initialPointCounting(uint32_t initialDepth) {
    itsGlobalOctreeDepth = initialDepth;

    // Precalculate parameters
    itsGlobalOctreeBase = static_cast<uint32_t >(pow(2, initialDepth));
    for(uint32_t gridSize = itsGlobalOctreeBase; gridSize > 0; gridSize >>= 1) {
        itsVoxelAmountDense += static_cast<uint32_t>(pow(gridSize, 3));
    }
    spdlog::info("The dense octree grid cell amount: {}", itsVoxelAmountDense);

    // Allocate the dense point count
    itsDensePointCountPerVoxel = make_unique<CudaArray<uint32_t>>(itsVoxelAmountDense, "itsDensePointCountPerVoxel");
    gpuErrchk(cudaMemset (itsDensePointCountPerVoxel->devicePointer(), 0, itsVoxelAmountDense * sizeof(uint32_t)));

    // Allocate the conversion LUT from dense to sparse
    itsDenseToSparseLUT = make_unique<CudaArray<int>>(itsVoxelAmountDense, "denseToSparseLUT");
    gpuErrchk(cudaMemset (itsDenseToSparseLUT->devicePointer(), -1, itsVoxelAmountDense * sizeof(int)));

    // Allocate the global sparseIndexCounter
    itsVoxelAmountSparse = make_unique<CudaArray<uint32_t>>(1, "sparseVoxelAmount");
    gpuErrchk(cudaMemset (itsVoxelAmountSparse->devicePointer(), 0, 1 * sizeof(uint32_t)));

    // Calculate kernel dimensions
    dim3 grid, block;
    tools::create1DKernel(block, grid, itsMetadata.pointAmount);

    // Initial point counting
    tools::KernelTimer timer;
    timer.start();
    kernelCountingSparse <<<  grid, block >>> (
                    itsCloudData->devicePointer(),
                    itsDensePointCountPerVoxel->devicePointer(),
                    itsDenseToSparseLUT->devicePointer(),
                    itsVoxelAmountSparse->devicePointer(),
                    itsMetadata,
                    itsGlobalOctreeBase);
    timer.stop();
    gpuErrchk(cudaGetLastError());

    itsTimeMeasurement.insert(std::make_pair("initialPointCount", timer.getMilliseconds()));
    spdlog::info("'initialPointCountingSparse' took {:f} [ms]", timer.getMilliseconds());
}