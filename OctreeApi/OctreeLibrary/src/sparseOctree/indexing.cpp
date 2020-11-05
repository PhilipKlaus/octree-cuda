//
// Created by KlausP on 04.11.2020.
//

#include <sparseOctree.h>
#include <tools.cuh>
#include <iostream>


void SparseOctree::hierarchicalCount(
        const unique_ptr<Chunk[]> &h_octreeSparse,
        const unique_ptr<int[]> &h_sparseToDenseLUT,
        uint32_t sparseVoxelIndex,
        uint32_t level) {

    Chunk voxel = h_octreeSparse[sparseVoxelIndex];

    // 1. Depth first traversal
    bool hasFinishedChildren = false;
    for(auto i = 0; i < voxel.childrenChunksCount; ++i) {
        if(h_octreeSparse[voxel.childrenChunks[i]].isFinished) {
            hasFinishedChildren = true;
            hierarchicalCount(h_octreeSparse, h_sparseToDenseLUT, voxel.childrenChunks[i], level - 1);
        }
    }

    // 2. Now we can assure that all direct childrens have subsamples
    if(hasFinishedChildren) {

        // 3. Calculate the dense voxel index
        auto denseVoxelIndex = h_sparseToDenseLUT[sparseVoxelIndex];

        // 4. Calculate the dense coordinates of the voxel
        BoundingBox bb{};
        Vector3i coords{};
        calculateVoxelBB(bb, coords, denseVoxelIndex, level);

        PointCloudMetadata metadata{};
        metadata.scale = itsMetadata.scale;
        metadata.boundingBox = bb;

        std::cout << "min: " << "x: " << metadata.boundingBox.minimum.x << ", y: " << metadata.boundingBox.minimum.y << ", z: " << metadata.boundingBox.minimum.z << std::endl;
        std::cout << "max: " << "x: " << metadata.boundingBox.maximum.x << ", y: " << metadata.boundingBox.maximum.y << ", z: " << metadata.boundingBox.maximum.z << std::endl;

        // 5. Spawn kernel for calculating
        uint32_t subsamplePointCount = 0;

        voxel.pointCount = subsamplePointCount;
    }
}

void SparseOctree::performIndexing() {

// 1. We can assume that each group of 8 child voxels form a continuous space within the 'itsDataLUT'
// 2. We can assume that the first child is childrenChunks[0] and that its 'chunkDataIndex' points to the beginning
//    of this continuous memory

    auto h_octreeSparse = itsOctreeSparse->toHost();
    auto h_sparseToDenseLUT = itsSparseToDenseLUT->toHost();
    uint32_t rootVoxelIndexSparse = itsVoxelAmountSparse->toHost()[0] - 1;

    hierarchicalCount(h_octreeSparse, h_sparseToDenseLUT, rootVoxelIndexSparse, itsGlobalOctreeDepth);

    /*dim3 grid, block;
    tools::create1DKernel(block, grid, newCellAmount);

    tools::KernelTimer timer;
    timer.start();
    kernelEvaluateSparseOctree << < grid, block >> > (
            itsDensePointCountPerVoxel->devicePointer(),
                    itsDenseToSparseLUT->devicePointer(),
                    itsVoxelAmountSparse->devicePointer(),
                    newCellAmount,
                    gridSize>>1,
                    gridSize,
                    cellOffsetNew,
                    cellOffsetOld);
    timer.stop();
    gpuErrchk(cudaGetLastError()); */
};

