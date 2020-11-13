//
// Created by KlausP on 04.11.2020.
//

#include <sparseOctree.h>
#include <chunking.cuh>

SparseOctree::SparseOctree(uint32_t depth, uint32_t mergingThreshold, PointCloudMetadata cloudMetadata, unique_ptr<CudaArray<Vector3>> cloudData) :
        itsCloudData(move(cloudData)),
        itsPointCloudMetadata(cloudMetadata)
{
    // Initialize octree metadata
    itsMetadata.depth = depth;
    itsMetadata.nodeAmountDense = 0;
    itsMetadata.nodeAmountSparse = 0;
    itsMetadata.mergingThreshold = mergingThreshold;

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

unique_ptr<uint32_t[]> SparseOctree::getDensePointCountPerVoxel() {
    return itsDensePointCountPerVoxel->toHost();
}

unique_ptr<int[]> SparseOctree::getDenseToSparseLUT() {
    return itsDenseToSparseLUT->toHost();
}

unique_ptr<int[]> SparseOctree::getSparseToDenseLUT(){
    return itsSparseToDenseLUT->toHost();
}

unique_ptr<Chunk[]> SparseOctree::getOctreeSparse() {
    return itsOctreeSparse->toHost();
}

uint32_t SparseOctree::getVoxelAmountSparse() {
    return itsMetadata.nodeAmountSparse;
}

unordered_map<uint32_t, unique_ptr<CudaArray<uint32_t>>> const& SparseOctree::getSubsampleLUT() const {
    return itsSubsampleLUTs;
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
            itsPointCloudMetadata,
            itsGridSideLengthPerLevel[0]);

    itsTimeMeasurement.insert(std::make_pair("distributePointsSparse", time));
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
            itsPointCloudMetadata,
            itsGridSideLengthPerLevel[0]
    );

    // Store the current amount of sparse nodes
    // !IMPORTANT! At this time the amount of sparse node just holds the amount of sparse nodes in the base level
    itsMetadata.nodeAmountSparse = nodeAmountSparse->toHost()[0];
    itsTimeMeasurement.insert(std::make_pair("initialPointCount", time));
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

        itsTimeMeasurement.insert(std::make_pair("EvaluateSparseOctree_" + std::to_string(itsGridSideLengthPerLevel[i]), time));
        timeAccumulated += time;
    }

    spdlog::info("'EvaluateSparseOctree' took {:f} [ms]", timeAccumulated);

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

    initializeBaseGridSparse();
    initializeOctreeSparse();
}


// ToDo: Rename
void SparseOctree::initializeBaseGridSparse() {

    float time = chunking::initLowestOctreeHierarchy(
            itsOctreeSparse,
            itsDensePointCountPerVoxel,
            itsDenseToSparseLUT,
            itsSparseToDenseLUT,
            itsVoxelsPerLevel[0]);

    itsTimeMeasurement.insert(std::make_pair("initializeBaseGridSparse", time));
}

// ToDo: Rename
void SparseOctree::initializeOctreeSparse() {

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

    spdlog::info("'initializeOctreeSparse' took {:f} [ms]", timeAccumulated);
}