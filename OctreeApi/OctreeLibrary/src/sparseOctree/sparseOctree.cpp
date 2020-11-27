//
// Created by KlausP on 04.11.2020.
//

#include <sparseOctree.h>
#include <chunking.cuh>
#include <pseudo_random_subsampling.cuh>

SparseOctree::SparseOctree(uint32_t depth, uint32_t mergingThreshold, PointCloudMetadata cloudMetadata, unique_ptr<CudaArray<uint8_t>> cloudData) :
        itsCloudData(move(cloudData))
{
    // Initialize octree metadata
    itsMetadata.depth = depth;
    itsMetadata.nodeAmountDense = 0;
    itsMetadata.nodeAmountSparse = 0;
    itsMetadata.leafNodeAmount = 0;
    itsMetadata.parentNodeAmount = 0;
    itsMetadata.meanPointsPerLeafNode = 0.f;
    itsMetadata.stdevPointsPerLeafNode = 0.f;
    itsMetadata.minPointsPerNode = 0;
    itsMetadata.maxPointsPerNode = 0;
    itsMetadata.mergingThreshold = mergingThreshold;
    itsMetadata.cloudMetadata = cloudMetadata;

    // Pre calculate often-used octree metrics
    auto sideLength = static_cast<uint32_t >(pow(2, depth));
    for(uint32_t gridSize = sideLength; gridSize > 0; gridSize >>= 1) {
        itsGridSideLengthPerLevel.push_back(gridSize);
        itsLinearizedDenseVoxelOffset.push_back(itsMetadata.nodeAmountDense);
        itsVoxelsPerLevel.push_back(static_cast<uint32_t>(pow(gridSize, 3)));
        itsMetadata.nodeAmountDense += static_cast<uint32_t>(pow(gridSize, 3));
    }

    // Create data LUT
    itsDataLUT = make_unique<CudaArray<uint32_t>>(cloudMetadata.pointAmount, "Data LUT");

    spdlog::info("Instantiated sparse octree for {} points", cloudMetadata.pointAmount);
}


//###################
//#     Pipeline    #
//###################

void SparseOctree::distributePoints() {

    // Create temporary indexRegister for assigning an index for each point within its chunk area
    auto tmpIndexRegister = make_unique<CudaArray<uint32_t>>(itsMetadata.nodeAmountSparse, "tmpIndexRegister");
    gpuErrchk(cudaMemset (tmpIndexRegister->devicePointer(), 0, itsMetadata.nodeAmountSparse * sizeof(uint32_t)));

    float time = chunking::distributePoints(
            itsOctreeSparse,
            itsCloudData,
            itsDataLUT,
            itsDenseToSparseLUT,
            tmpIndexRegister,
            itsMetadata.cloudMetadata,
            itsGridSideLengthPerLevel[0]);

    itsTimeMeasurement.insert(std::make_pair("distributePointsSparse", time));
    spdlog::info("'distributePoints' took {:f} [ms]", time);
}

void SparseOctree::initialPointCounting() {

    // Allocate the dense point count
    itsDensePointCountPerVoxel = make_unique<CudaArray<uint32_t>>(itsMetadata.nodeAmountDense, "itsDensePointCountPerVoxel");
    gpuErrchk(cudaMemset (itsDensePointCountPerVoxel->devicePointer(), 0, itsMetadata.nodeAmountDense * sizeof(uint32_t)));

    // Allocate the conversion LUT from dense to sparse
    itsDenseToSparseLUT = make_unique<CudaArray<int>>(itsMetadata.nodeAmountDense, "denseToSparseLUT");
    gpuErrchk(cudaMemset (itsDenseToSparseLUT->devicePointer(), -1, itsMetadata.nodeAmountDense * sizeof(int)));

    // Allocate the temporary sparseIndexCounter
    auto nodeAmountSparse = make_unique<CudaArray<uint32_t>>(1, "nodeAmountSparse");
    gpuErrchk(cudaMemset (nodeAmountSparse->devicePointer(), 0, 1 * sizeof(uint32_t)));

    float time = chunking::initialPointCounting(
            itsCloudData,
            itsDensePointCountPerVoxel,
            itsDenseToSparseLUT,
            nodeAmountSparse,
            itsMetadata.cloudMetadata,
            itsGridSideLengthPerLevel[0]
    );

    // Store the current amount of sparse nodes
    // !IMPORTANT! At this time the amount of sparse node just holds the amount of sparse nodes in the base level
    itsMetadata.nodeAmountSparse = nodeAmountSparse->toHost()[0];
    itsTimeMeasurement.insert(std::make_pair("initialPointCount", time));
    spdlog::info("'initialPointCounting' took {:f} [ms]", time);
}
void SparseOctree::performCellMerging() {

    // Allocate the temporary sparseIndexCounter
    auto nodeAmountSparse = make_unique<CudaArray<uint32_t>>(1, "nodeAmountSparse");
    // !IMPORTANT! initialize it with the current sparse node counts (from base level)
    gpuErrchk(cudaMemcpy(nodeAmountSparse->devicePointer(), &itsMetadata.nodeAmountSparse, 1 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    float timeAccumulated = 0;

    // Perform a hierarchicaly merging of the grid cells which results in an octree structure
    for(uint32_t i = 0; i < itsMetadata.depth; ++i) {

        float time = chunking::propagatePointCounts (
                itsDensePointCountPerVoxel,
                itsDenseToSparseLUT,
                nodeAmountSparse,
                itsVoxelsPerLevel[i + 1],
                itsGridSideLengthPerLevel[i + 1],
                itsGridSideLengthPerLevel[i],
                itsLinearizedDenseVoxelOffset[i + 1],
                itsLinearizedDenseVoxelOffset[i]);

        itsTimeMeasurement.insert(std::make_pair("PropagatePointCounts_" + std::to_string(itsGridSideLengthPerLevel[i]), time));
        timeAccumulated += time;
    }

    spdlog::info("'propagatePointCounts' took {:f} [ms]", timeAccumulated);

    // Retrieve the actual amount of sparse nodes in the octree and allocate the octree data structure
    itsMetadata.nodeAmountSparse = nodeAmountSparse->toHost()[0];
    itsOctreeSparse = make_unique<CudaArray<Chunk>>(itsMetadata.nodeAmountSparse, "octreeSparse");

    spdlog::info(
            "Sparse octree ({} voxels) -> Memory saving: {:f} [%] {:f} [GB]",
            itsMetadata.nodeAmountSparse,
            (1 - static_cast<float>(itsMetadata.nodeAmountSparse) / itsMetadata.nodeAmountDense) * 100,
            static_cast<float>(itsMetadata.nodeAmountDense - itsMetadata.nodeAmountSparse) * sizeof(Chunk) / 1000000000.f
    );

    // Allocate the conversion LUT from sparse to dense
    itsSparseToDenseLUT = make_unique<CudaArray<int>>(itsMetadata.nodeAmountSparse, "sparseToDenseLUT");
    gpuErrchk(cudaMemset (itsSparseToDenseLUT->devicePointer(), -1, itsMetadata.nodeAmountSparse * sizeof(int)));

    initLowestOctreeHierarchy();
    mergeHierarchical();
}


// ToDo: Rename
void SparseOctree::initLowestOctreeHierarchy() {

    float time = chunking::initLowestOctreeHierarchy(
            itsOctreeSparse,
            itsDensePointCountPerVoxel,
            itsDenseToSparseLUT,
            itsSparseToDenseLUT,
            itsVoxelsPerLevel[0]);

    itsTimeMeasurement.insert(std::make_pair("initLowestOctreeHierarchy", time));
    spdlog::info("'initLowestOctreeHierarchy' took {:f} [ms]", time);
}

void SparseOctree::mergeHierarchical() {

    // Create a temporary counter register for assigning indices for chunks within the 'itsDataLUT' register
    auto globalChunkCounter = make_unique<CudaArray<uint32_t>>(1, "globalChunkCounter");
    gpuErrchk(cudaMemset (globalChunkCounter->devicePointer(), 0, 1 * sizeof(uint32_t)));

    // Perform a hierarchicaly merging of the grid cells which results in an octree structure
    float timeAccumulated = 0;
    for(uint32_t i = 0; i < itsMetadata.depth; ++i) {

        float time = chunking::mergeHierarchical(
                itsOctreeSparse,
                itsDensePointCountPerVoxel,
                itsDenseToSparseLUT,
                itsSparseToDenseLUT,
                globalChunkCounter,
                itsMetadata.mergingThreshold,
                itsVoxelsPerLevel[i + 1],
                itsGridSideLengthPerLevel[i + 1],
                itsGridSideLengthPerLevel[i],
                itsLinearizedDenseVoxelOffset[i + 1],
                itsLinearizedDenseVoxelOffset[i]
        );

        timeAccumulated += time;
        itsTimeMeasurement.insert(std::make_pair("initializeOctreeSparse_" + std::to_string(itsGridSideLengthPerLevel[i]), time));
    }

    spdlog::info("'mergeHierarchical' took {:f} [ms]", timeAccumulated);
}

float SparseOctree::hierarchicalSubsampling(
        const unique_ptr<Chunk[]> &h_octreeSparse,
        const unique_ptr<int[]> &h_sparseToDenseLUT,
        uint32_t sparseVoxelIndex,
        uint32_t level,
        unique_ptr<CudaArray<uint32_t>> &subsampleCountingGrid,
        unique_ptr<CudaArray<int>> &subsampleDenseToSparseLUT,
        unique_ptr<CudaArray<uint32_t>> &subsampleSparseVoxelCount) {

    Chunk voxel = h_octreeSparse[sparseVoxelIndex];
    float accumulatedTime = 0;

    // 1. Depth first traversal
    for(uint32_t i = 0; i < voxel.childrenChunksCount; ++i) {
        if(h_octreeSparse[voxel.childrenChunks[i]].isFinished) {
            accumulatedTime += hierarchicalSubsampling(
                    h_octreeSparse,
                    h_sparseToDenseLUT,
                    voxel.childrenChunks[i],
                    level - 1,
                    subsampleCountingGrid,
                    subsampleDenseToSparseLUT,
                    subsampleSparseVoxelCount);
        }
    }

    // 2. Now we can assure that all direct children have subsamples
    if(voxel.isParent) {

        // 3. Calculate the dense coordinates of the voxel
        BoundingBox bb{};
        Vector3i coords{};
        auto denseVoxelIndex = h_sparseToDenseLUT[sparseVoxelIndex];
        calculateVoxelBB(bb, coords, itsMetadata.cloudMetadata.boundingBox, denseVoxelIndex, level);

        PointCloudMetadata metadata = itsMetadata.cloudMetadata;
        metadata.boundingBox = bb;
        metadata.cloudOffset = bb.minimum;

        gpuErrchk(cudaMemset (subsampleCountingGrid->devicePointer(), 0, subsampleCountingGrid->pointCount() * sizeof(uint32_t)));
        gpuErrchk(cudaMemset (subsampleDenseToSparseLUT->devicePointer(), 0, subsampleDenseToSparseLUT->pointCount() * sizeof(uint32_t)));
        gpuErrchk(cudaMemset (subsampleSparseVoxelCount->devicePointer(), 0, 1 * sizeof(uint32_t)));

        // 4. Pre-calculate the subsamples and count the subsampled points
        for(uint32_t i = 0; i < voxel.childrenChunksCount; ++i) {

            uint32_t childIndex = voxel.childrenChunks[i];
            Chunk child = h_octreeSparse[childIndex];
            metadata.pointAmount = child.isParent ? itsSubsampleLUTs[childIndex]->pointCount() : child.pointCount;

            accumulatedTime += pseudo__random_subsampling::subsample(
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
        gpuErrchk(cudaMemset (subsampleCountingGrid->devicePointer(), 0, subsampleCountingGrid->pointCount() * sizeof(uint32_t)));
        for(uint32_t i = 0; i < voxel.childrenChunksCount; ++i) {

            uint32_t childIndex = voxel.childrenChunks[i];
            Chunk child = h_octreeSparse[childIndex];
            metadata.pointAmount = child.isParent ? itsSubsampleLUTs[childIndex]->pointCount() : child.pointCount;

            accumulatedTime += pseudo__random_subsampling::distributeSubsamples(
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
    return accumulatedTime;
}

void SparseOctree::performSubsampling() {

    auto h_octreeSparse = itsOctreeSparse->toHost();
    auto h_sparseToDenseLUT = itsSparseToDenseLUT->toHost();
    uint32_t rootVoxelIndexSparse = itsMetadata.nodeAmountSparse - 1;

    // Prepare data strucutres for the subsampling
    auto pointCountGrid = make_unique<CudaArray<uint32_t >>(itsVoxelsPerLevel[0], "pointCountGrid");
    auto denseToSpareLUT = make_unique<CudaArray<int >>(itsVoxelsPerLevel[0], "denseToSpareLUT");
    auto voxelCount = make_unique<CudaArray<uint32_t >>(1, "voxelCount");

    // Perform the actual subsampling
    float time = hierarchicalSubsampling(
            h_octreeSparse,
            h_sparseToDenseLUT,
            rootVoxelIndexSparse,
            itsMetadata.depth,
            pointCountGrid,
            denseToSpareLUT,
            voxelCount
    );

    spdlog::info("subsampling took {}[ms]", time);
}

