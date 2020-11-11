#include <sparseOctree.h>
#include "tools.cuh"
#include "timing.cuh"
#include "defines.cuh"


__global__ void kernelInitializeBaseGridSparse(
        Chunk *octreeSparse,
        uint32_t *densePointCount,
        int *denseToSparseLUT,
        int *sparseToDenseLUT,
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

    // Update sparseToDense LUT
    sparseToDenseLUT[sparseVoxelIndex] = denseVoxelIndex;

    Chunk *chunk = octreeSparse + sparseVoxelIndex;
    chunk->pointCount = densePointCount[denseVoxelIndex];
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

void SparseOctree::performCellMerging(uint32_t threshold) {

    float time = 0;

    // Perform a hierarchicaly merging of the grid cells which results in an octree structure
    for(uint32_t i = 0; i < itsGlobalOctreeDepth; ++i) {

        dim3 grid, block;
        tools::create1DKernel(block, grid, itsVoxelsPerLevel[i + 1]);

        tools::KernelTimer timer;
        timer.start();
        kernelEvaluateSparseOctree << < grid, block >> > (
                itsDensePointCountPerVoxel->devicePointer(),
                        itsDenseToSparseLUT->devicePointer(),
                        itsVoxelAmountSparse->devicePointer(),
                        itsVoxelsPerLevel[i + 1],
                        itsGridSideLengthPerLevel[i + 1],
                        itsGridSideLengthPerLevel[i],
                        itsLinearizedDenseVoxelOffset[i + 1],
                        itsLinearizedDenseVoxelOffset[i]);
        timer.stop();
        gpuErrchk(cudaGetLastError());

        time += timer.getMilliseconds();
        itsTimeMeasurement.insert(std::make_pair("EvaluateSparseOctree_" + std::to_string(itsGridSideLengthPerLevel[i]), timer.getMilliseconds()));
    }

    spdlog::info("'EvaluateSparseOctree' took {:f} [ms]", time);

    // Create the sparse octree
    uint32_t voxelAmountSparse = itsVoxelAmountSparse->toHost()[0];
    itsOctreeSparse = make_unique<CudaArray<Chunk>>(voxelAmountSparse, "octreeSparse");

    spdlog::info(
            "Sparse octree ({} voxels) -> Memory saving: {:f} [%] {:f} [GB]",
            voxelAmountSparse,
            (1 - static_cast<float>(voxelAmountSparse) / itsVoxelAmountDense) * 100,
            static_cast<float>(itsVoxelAmountDense - voxelAmountSparse) * sizeof(Chunk) / 1000000000.f
    );

    // Allocate the conversion LUT from sparse to dense
    itsSparseToDenseLUT = make_unique<CudaArray<int>>(voxelAmountSparse, "sparseToDenseLUT");
    gpuErrchk(cudaMemset (itsSparseToDenseLUT->devicePointer(), -1, voxelAmountSparse * sizeof(int)));

    initializeBaseGridSparse();
    initializeOctreeSparse(threshold);
}

void SparseOctree::initializeBaseGridSparse() {

    dim3 grid, block;
    tools::create1DKernel(block, grid, itsVoxelsPerLevel[0]);

    tools::KernelTimer timer;
    timer.start();
    kernelInitializeBaseGridSparse << < grid, block >> > (
            itsOctreeSparse->devicePointer(),
            itsDensePointCountPerVoxel->devicePointer(),
            itsDenseToSparseLUT->devicePointer(),
            itsSparseToDenseLUT->devicePointer(),
            itsVoxelsPerLevel[0]);
    timer.stop();
    gpuErrchk(cudaGetLastError());

    itsTimeMeasurement.insert(std::make_pair("initializeBaseGridSparse", timer.getMilliseconds()));
    spdlog::info("'initializeBaseGridSparse' took {:f} [ms]", timer.getMilliseconds());
}
