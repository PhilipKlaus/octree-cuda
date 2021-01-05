//
// Created by KlausP on 04.11.2020.
//

#include <sparseOctree.h>

// Includes for chunking
#include <point_counting.cuh>
#include <octree_initialization.cuh>
#include <point_count_propagation.cuh>
#include <hierarchical_merging.cuh>
#include <point_distributing.cuh>


SparseOctree::SparseOctree(
        GridSize chunkingGrid,
        GridSize subsamplingGrid,
        uint32_t mergingThreshold,
        PointCloudMetadata cloudMetadata,
        unique_ptr<CudaArray<uint8_t>> cloudData,
        SubsamplingStrategy strategy) :

        itsCloudData(move(cloudData))
{
    // Initialize octree metadata
    itsMetadata = {};
    itsMetadata.depth = tools::getOctreeLevel(chunkingGrid);
    itsMetadata.chunkingGrid = chunkingGrid;
    itsMetadata.subsamplingGrid = subsamplingGrid;
    itsMetadata.mergingThreshold = mergingThreshold;
    itsMetadata.cloudMetadata = cloudMetadata;
    itsMetadata.strategy = strategy;

    // Pre calculate often-used octree metrics
    auto sideLength = static_cast<uint32_t >(pow(2, itsMetadata.depth));
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

    float time = chunking::initialPointCounting<float>(
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
    itsTimeMeasurement.emplace_back("initialPointCount", time);
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

        itsTimeMeasurement.emplace_back("propagatePointCounts_" + std::to_string(itsGridSideLengthPerLevel[i]), time);
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


void SparseOctree::initLowestOctreeHierarchy() {

    float time = chunking::initOctree(
            itsOctreeSparse,
            itsDensePointCountPerVoxel,
            itsDenseToSparseLUT,
            itsSparseToDenseLUT,
            itsVoxelsPerLevel[0]);

    itsTimeMeasurement.emplace_back("initLowestOctreeHierarchy", time);
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
        itsTimeMeasurement.emplace_back("mergeHierarchical_" + std::to_string(itsGridSideLengthPerLevel[i]), time);
    }

    spdlog::info("'mergeHierarchical' took {:f} [ms]", timeAccumulated);
}


void SparseOctree::distributePoints() {

    // Create temporary indexRegister for assigning an index for each point within its chunk area
    auto tmpIndexRegister = make_unique<CudaArray<uint32_t>>(itsMetadata.nodeAmountSparse, "tmpIndexRegister");
    gpuErrchk(cudaMemset (tmpIndexRegister->devicePointer(), 0, itsMetadata.nodeAmountSparse * sizeof(uint32_t)));

    float time = chunking::distributePoints<float>(
            itsOctreeSparse,
            itsCloudData,
            itsDataLUT,
            itsDenseToSparseLUT,
            tmpIndexRegister,
            itsMetadata.cloudMetadata,
            itsGridSideLengthPerLevel[0]);

    itsTimeMeasurement.emplace_back("distributePointsSparse", time);
    spdlog::info("'distributePoints' took {:f} [ms]", time);
}


void SparseOctree::performSubsampling() {

    auto h_octreeSparse = itsOctreeSparse->toHost();
    auto h_sparseToDenseLUT = itsSparseToDenseLUT->toHost();
    auto nodesBaseLevel = static_cast<uint32_t>(pow(itsMetadata.subsamplingGrid, 3.f));

    // Prepare data strucutres for the subsampling
    auto pointCountGrid = make_unique<CudaArray<uint32_t >>(nodesBaseLevel, "pointCountGrid");
    auto denseToSpareLUT = make_unique<CudaArray<int >>(nodesBaseLevel, "denseToSpareLUT");
    auto voxelCount = make_unique<CudaArray<uint32_t >>(1, "voxelCount");

    gpuErrchk(cudaMemset (pointCountGrid->devicePointer(), 0, pointCountGrid->pointCount() * sizeof(uint32_t)));
    gpuErrchk(cudaMemset (denseToSpareLUT->devicePointer(), -1, denseToSpareLUT->pointCount() * sizeof(uint32_t)));
    gpuErrchk(cudaMemset (voxelCount->devicePointer(), 0, 1 * sizeof(uint32_t)));

    std::tuple<float, float> time;

    if(itsMetadata.strategy == RANDOM_POINT) {
        auto randomStates = make_unique<CudaArray<curandState_t >>(1024, "randomStates");
        auto randomIndices = make_unique<CudaArray<uint32_t >>(nodesBaseLevel, "randomIndices");

        time = randomSubsampling(
                h_octreeSparse,
                h_sparseToDenseLUT,
                getRootIndex(),
                itsMetadata.depth,
                pointCountGrid,
                denseToSpareLUT,
                voxelCount,
                randomStates,
                randomIndices);
    }
    else {
        time = firstPointSubsampling(
                h_octreeSparse,
                h_sparseToDenseLUT,
                getRootIndex(),
                itsMetadata.depth,
                pointCountGrid,
                denseToSpareLUT,
                voxelCount);
    }


    itsTimeMeasurement.emplace_back("subsampleEvaluation", get<0>(time));
    itsTimeMeasurement.emplace_back("subsampling", get<1>(time));
    spdlog::info("subsample evaluation took {}[ms]", get<0>(time));
    spdlog::info("subsampling took {}[ms]", get<1>(time));
}

