//
// Created by KlausP on 04.11.2020.
//

#include <sparseOctree.h>
#include <chunking.cuh>

SparseOctree::SparseOctree(PointCloudMetadata cloudMetadata, unique_ptr<CudaArray<Vector3>> cloudData) :
itsCloudData(move(cloudData)),
itsMetadata(cloudMetadata),
itsGlobalOctreeDepth(0),
itsVoxelAmountDense(0)
{
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
    return itsVoxelAmountSparse->toHost()[0];
}

unordered_map<uint32_t, unique_ptr<CudaArray<uint32_t>>> const& SparseOctree::getSubsampleLUT() const {
    return itsSubsampleLUTs;
}

//###################
//#     Pipeline    #
//###################

void SparseOctree::distributePoints() {

    // Create temporary indexRegister for assigning an index for each point within its chunk area
    auto cellAmountSparse = itsVoxelAmountSparse->toHost()[0];
    auto tmpIndexRegister = make_unique<CudaArray<uint32_t>>(cellAmountSparse, "tmpIndexRegister");
    gpuErrchk(cudaMemset (tmpIndexRegister->devicePointer(), 0, cellAmountSparse * sizeof(uint32_t)));

    float time = chunking::distributePoints(
            itsOctreeSparse,
            itsCloudData,
            itsDataLUT,
            itsDenseToSparseLUT,
            tmpIndexRegister,
            itsMetadata,
            itsGridSideLengthPerLevel[0]);

    itsTimeMeasurement.insert(std::make_pair("distributePointsSparse", time));
}

void SparseOctree::initialPointCounting(uint32_t initialDepth) {

    // Pre-calculate different Octree parameters
    preCalculateOctreeParameters(initialDepth);

    // Allocate the dense point count
    itsDensePointCountPerVoxel = make_unique<CudaArray<uint32_t>>(itsVoxelAmountDense, "itsDensePointCountPerVoxel");
    gpuErrchk(cudaMemset (itsDensePointCountPerVoxel->devicePointer(), 0, itsVoxelAmountDense * sizeof(uint32_t)));

    // Allocate the conversion LUT from dense to sparse
    itsDenseToSparseLUT = make_unique<CudaArray<int>>(itsVoxelAmountDense, "denseToSparseLUT");
    gpuErrchk(cudaMemset (itsDenseToSparseLUT->devicePointer(), -1, itsVoxelAmountDense * sizeof(int)));

    // Allocate the global sparseIndexCounter
    itsVoxelAmountSparse = make_unique<CudaArray<uint32_t>>(1, "sparseVoxelAmount");
    gpuErrchk(cudaMemset (itsVoxelAmountSparse->devicePointer(), 0, 1 * sizeof(uint32_t)));

    float time = chunking::initialPointCounting(
            itsCloudData,
            itsDensePointCountPerVoxel,
            itsDenseToSparseLUT,
            itsVoxelAmountSparse,
            itsMetadata,
            itsGridSideLengthPerLevel[0]
    );

    itsTimeMeasurement.insert(std::make_pair("initialPointCount", time));
}

void SparseOctree::performCellMerging(uint32_t threshold) {

    float timeAccumulated = 0;

    // Perform a hierarchicaly merging of the grid cells which results in an octree structure
    for(uint32_t i = 0; i < itsGlobalOctreeDepth; ++i) {

        float time = chunking::propagatePointCounts (
                itsDensePointCountPerVoxel,
                itsDenseToSparseLUT,
                itsVoxelAmountSparse,
                itsVoxelsPerLevel[i + 1],
                itsGridSideLengthPerLevel[i + 1],
                itsGridSideLengthPerLevel[i],
                itsLinearizedDenseVoxelOffset[i + 1],
                itsLinearizedDenseVoxelOffset[i]);

        itsTimeMeasurement.insert(std::make_pair("EvaluateSparseOctree_" + std::to_string(itsGridSideLengthPerLevel[i]), time));
        timeAccumulated += time;
    }

    spdlog::info("'EvaluateSparseOctree' took {:f} [ms]", timeAccumulated);

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
void SparseOctree::initializeOctreeSparse(uint32_t threshold) {

    // Create a temporary counter register for assigning indices for chunks within the 'itsDataLUT' register
    auto globalChunkCounter = make_unique<CudaArray<uint32_t>>(1, "globalChunkCounter");
    gpuErrchk(cudaMemset (globalChunkCounter->devicePointer(), 0, 1 * sizeof(uint32_t)));

    // Perform a hierarchicaly merging of the grid cells which results in an octree structure
    float timeAccumulated = 0;
    for(uint32_t i = 0; i < itsGlobalOctreeDepth; ++i) {

        float time = chunking::mergeHierarchical(
                itsOctreeSparse,
                itsDensePointCountPerVoxel,
                itsDenseToSparseLUT,
                itsSparseToDenseLUT,
                globalChunkCounter,
                threshold,
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