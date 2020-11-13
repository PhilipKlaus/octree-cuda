//
// Created by KlausP on 04.11.2020.
//

#include <sparseOctree.h>
#include <pseudo_random_subsampling.cuh>


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
        calculateVoxelBB(bb, coords, itsPointCloudMetadata.boundingBox, denseVoxelIndex, level);

        PointCloudMetadata metadata{};
        metadata.scale = itsPointCloudMetadata.scale;
        metadata.boundingBox = bb;
        metadata.cloudOffset = bb.minimum;

        gpuErrchk(cudaMemset (subsampleCountingGrid->devicePointer(), 0, itsVoxelsPerLevel[0] * sizeof(uint32_t)));
        gpuErrchk(cudaMemset (subsampleDenseToSparseLUT->devicePointer(), 0, itsVoxelsPerLevel[0] * sizeof(uint32_t)));
        gpuErrchk(cudaMemset (subsampleSparseVoxelCount->devicePointer(), 0, 1 * sizeof(uint32_t)));

        // 4. Pre-calculate the subsamples and count the subsampled points
        for(uint32_t i = 0; i < voxel.childrenChunksCount; ++i) {

            uint32_t childIndex = voxel.childrenChunks[i];
            Chunk child = h_octreeSparse[childIndex];
            metadata.pointAmount = child.isParent ? itsSubsampleLUTs[childIndex]->pointCount() : child.pointCount;

            float time = pseudo__random_subsampling::subsample(
                    itsCloudData,
                    child.isParent ? itsSubsampleLUTs[childIndex] : itsDataLUT,
                    child.isParent ? 0 : child.chunkDataIndex,
                    subsampleCountingGrid,
                    subsampleDenseToSparseLUT,
                    subsampleSparseVoxelCount,
                    metadata,
                    itsGridSideLengthPerLevel[0]
            );
        }

        // 5. Reserve memory for a data LUT for the parent node
        auto amountUsedVoxels = subsampleSparseVoxelCount->toHost()[0];
        auto subsampleLUT = make_unique<CudaArray<uint32_t >>(amountUsedVoxels, "subsampleLUT_" + to_string(sparseVoxelIndex));
        itsSubsampleLUTs.insert(make_pair(sparseVoxelIndex, move(subsampleLUT)));

        // 6. Distribute points to the parent data LUT
        // ToDo: Remove memset by different strategy
        gpuErrchk(cudaMemset (subsampleCountingGrid->devicePointer(), 0, itsVoxelsPerLevel[0] * sizeof(uint32_t)));
        for(uint32_t i = 0; i < voxel.childrenChunksCount; ++i) {

            uint32_t childIndex = voxel.childrenChunks[i];
            Chunk child = h_octreeSparse[childIndex];
            metadata.pointAmount = child.isParent ? itsSubsampleLUTs[childIndex]->pointCount() : child.pointCount;

            float time = pseudo__random_subsampling::distributeSubsamples(
                    itsCloudData,
                    child.isParent ? itsSubsampleLUTs[childIndex] : itsDataLUT,
                    child.isParent ? 0: child.chunkDataIndex,
                    itsSubsampleLUTs[sparseVoxelIndex],
                    subsampleCountingGrid,
                    subsampleDenseToSparseLUT,
                    metadata,
                    itsGridSideLengthPerLevel[0]
            );
        }
    }
}

void SparseOctree::performIndexing() {

    auto h_octreeSparse = itsOctreeSparse->toHost();
    auto h_sparseToDenseLUT = itsSparseToDenseLUT->toHost();
    uint32_t rootVoxelIndexSparse = itsVoxelAmountSparse->toHost()[0] - 1;

    // Prepare data strucutres for the subsampling
    auto pointCountGrid = make_unique<CudaArray<uint32_t >>(itsVoxelsPerLevel[0], "pointCountGrid");
    auto denseToSpareLUT = make_unique<CudaArray<int >>(itsVoxelsPerLevel[0], "denseToSpareLUT");
    auto voxelCount = make_unique<CudaArray<uint32_t >>(1, "voxelCount");

    // Perform the actual subsampling
    hierarchicalCount(
            h_octreeSparse,
            h_sparseToDenseLUT,
            rootVoxelIndexSparse,
            itsMetadata.depth,
            pointCountGrid,
            denseToSpareLUT,
            voxelCount
            );
};

