//
// Created by KlausP on 04.11.2020.
//

#include <sparseOctree.h>
#include <iostream>
#include <kernels.cuh>
#include <tools.cuh>
#include <timing.cuh>

// Move point indices from old (child LUT) to new (parent LUT)
__global__ void distributeSubsamples(
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

    // 1. Calculate the index within the dense grid
    auto denseVoxelIndex = tools::calculateGridIndex(point, metadata, gridSideLength);

    // 2. Accumulate the counter within the dense cell
    auto oldIndex = atomicAdd((countingGrid + denseVoxelIndex), 1);

    // 3. If the thread is the first one accumulating the counter within the cell -> move index from child to parent LUT
    if(oldIndex == 0) {
        parentDataLUT[denseToSparseLUT[denseVoxelIndex]] = childDataLUT[childDataLUTStart + index];
    }
}

float executeKernelDistributeSubsamples(
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
    distributeSubsamples <<<  grid, block >>> (
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

    spdlog::info("'kernelDistributeSubsamples' took {:f} [ms]", timer.getMilliseconds());
    return timer.getMilliseconds();
}

void SparseOctree::evaluateOctreeStatistics(const unique_ptr<Chunk[]> &h_octreeSparse, uint32_t sparseVoxelIndex) {
    Chunk chunk = h_octreeSparse[sparseVoxelIndex];

    ++itsRelevantNodes;

    // Leaf node
    if(!chunk.isParent) {
        ++itsLeafeNodes;
    }

    // No leaf node but relevant
    else {
        for(uint32_t i = 0; i < chunk.childrenChunksCount; ++i) {
            evaluateOctreeStatistics(h_octreeSparse, chunk.childrenChunks[i]);
        }
    }
}

void SparseOctree::hierarchicalCount(
        const unique_ptr<Chunk[]> &h_octreeSparse,
        const unique_ptr<int[]> &h_sparseToDenseLUT,
        uint32_t sparseVoxelIndex,
        uint32_t level,
        unique_ptr<CudaArray<uint32_t>> &subsampleCountingGrid,
        unique_ptr<CudaArray<int>> &subsampleDenseToSparseLUT,
        unique_ptr<CudaArray<uint32_t>> &subsampleSparseVoxelCount) {

    Chunk voxel = h_octreeSparse[sparseVoxelIndex];

    // 1. Depth first traversal
    for(uint32_t i = 0; i < voxel.childrenChunksCount; ++i) {
        if(h_octreeSparse[voxel.childrenChunks[i]].isFinished) {
            hierarchicalCount(h_octreeSparse, h_sparseToDenseLUT, voxel.childrenChunks[i], level - 1, subsampleCountingGrid, subsampleDenseToSparseLUT, subsampleSparseVoxelCount);
        }
    }

    // 2. Now we can assure that all direct children have subsamples
    if(voxel.isParent) {

        // 3. Calculate the dense coordinates of the voxel
        BoundingBox bb{};
        Vector3i coords{};
        auto denseVoxelIndex = h_sparseToDenseLUT[sparseVoxelIndex];
        calculateVoxelBB(bb, coords, itsMetadata.boundingBox, denseVoxelIndex, level);

        PointCloudMetadata metadata{};
        metadata.scale = itsMetadata.scale;
        metadata.boundingBox = bb;
        metadata.cloudOffset = bb.minimum;

        gpuErrchk(cudaMemset (subsampleCountingGrid->devicePointer(), 0, itsVoxelsPerLevel[0] * sizeof(uint32_t)));
        gpuErrchk(cudaMemset (subsampleDenseToSparseLUT->devicePointer(), 0, itsVoxelsPerLevel[0] * sizeof(uint32_t)));
        gpuErrchk(cudaMemset (subsampleSparseVoxelCount->devicePointer(), 0, 1 * sizeof(uint32_t)));

        // 4. Pre-calculate the subsamples and count the subsampled points
        for(uint32_t i = 0; i < voxel.childrenChunksCount; ++i) {

            Chunk child = h_octreeSparse[voxel.childrenChunks[i]];

            if(!child.isParent) {

                metadata.pointAmount = child.pointCount;

                float time = kernelExecution::executeKernelMapCloudToGrid_LUT(
                        itsCloudData,
                        itsDataLUT,
                        child.chunkDataIndex,
                        subsampleCountingGrid,
                        subsampleDenseToSparseLUT,
                        subsampleSparseVoxelCount,
                        metadata,
                        itsGridSideLengthPerLevel[0]
                );

            }
            else {

                metadata.pointAmount = itsSubsampleLUTs[voxel.childrenChunks[i]]->pointCount();

                float time = kernelExecution::executeKernelMapCloudToGrid_LUT(
                        itsCloudData,
                        itsSubsampleLUTs[voxel.childrenChunks[i]],
                        0,
                        subsampleCountingGrid,
                        subsampleDenseToSparseLUT,
                        subsampleSparseVoxelCount,
                        metadata,
                        itsGridSideLengthPerLevel[0]
                );

            }
        }

        // 5. Reserve memory for a data LUT for the parent node
        auto amountUsedVoxels = subsampleSparseVoxelCount->toHost()[0];
        auto subsampleLUT = make_unique<CudaArray<uint32_t >>(amountUsedVoxels, "subsampleLUT_" + to_string(sparseVoxelIndex));
        itsSubsampleLUTs.insert(make_pair(sparseVoxelIndex, move(subsampleLUT)));

        // 6. Distribute points to the parent data LUT
        // ToDo: Remove memset by different strategy
        gpuErrchk(cudaMemset (subsampleCountingGrid->devicePointer(), 0, itsVoxelsPerLevel[0] * sizeof(uint32_t)));
        for(uint32_t i = 0; i < voxel.childrenChunksCount; ++i) {

            Chunk child = h_octreeSparse[voxel.childrenChunks[i]];

            if(!child.isParent) {

                metadata.pointAmount = child.pointCount;

                float time = executeKernelDistributeSubsamples(
                        itsCloudData,
                        itsDataLUT,
                        child.chunkDataIndex,
                        itsSubsampleLUTs[sparseVoxelIndex],
                        subsampleCountingGrid,
                        subsampleDenseToSparseLUT,
                        metadata,
                        itsGridSideLengthPerLevel[0]
                );
            }
            else {
                metadata.pointAmount = itsSubsampleLUTs[voxel.childrenChunks[i]]->pointCount();

                float time = executeKernelDistributeSubsamples(
                        itsCloudData,
                        itsSubsampleLUTs[voxel.childrenChunks[i]],
                        0,
                        itsSubsampleLUTs[sparseVoxelIndex],
                        subsampleCountingGrid,
                        subsampleDenseToSparseLUT,
                        metadata,
                        itsGridSideLengthPerLevel[0]
                );
            }
        }

        spdlog::info("level [{}]: Subsampled {} voxels into {} voxels", level, itsVoxelsPerLevel[0], amountUsedVoxels);
    }
}

void SparseOctree::performIndexing() {

    auto h_octreeSparse = itsOctreeSparse->toHost();
    auto h_sparseToDenseLUT = itsSparseToDenseLUT->toHost();
    uint32_t rootVoxelIndexSparse = itsVoxelAmountSparse->toHost()[0] - 1;

    // Gather some octree statistics
    evaluateOctreeStatistics(h_octreeSparse, rootVoxelIndexSparse);

    // Prepare data strucutres for the subsampling
    auto pointCountGrid = make_unique<CudaArray<uint32_t >>(itsVoxelsPerLevel[0], "pointCountGrid");
    auto denseToSpareLUT = make_unique<CudaArray<int >>(itsVoxelsPerLevel[0], "denseToSpareLUT");
    auto voxelCount = make_unique<CudaArray<uint32_t >>(1, "voxelCount");

    // Perform the actual subsampling
    hierarchicalCount(h_octreeSparse, h_sparseToDenseLUT, rootVoxelIndexSparse, itsGlobalOctreeDepth, pointCountGrid, denseToSpareLUT, voxelCount);

    // Print some octree statistics
    spdlog::error("Sparse octree overall nodes: {}", itsVoxelAmountSparse->toHost()[0]);
    spdlog::error("Sparse octree relevant nodes: {}", itsRelevantNodes);
    spdlog::error("Sparse octree leaf nodes: {}", itsLeafeNodes);
    spdlog::error("Sparse octree non-leaf nodes: {}", itsRelevantNodes-itsLeafeNodes);

};

