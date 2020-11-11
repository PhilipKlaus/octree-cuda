//
// Created by KlausP on 04.11.2020.
//

#include <sparseOctree.h>
#include <chunking.cuh>
uint32_t SparseOctree::exportTreeNode(
        Vector3 *cpuPointCloud,
        const unique_ptr<Chunk[]> &octreeSparse,
        const unique_ptr<uint32_t[]> &dataLUT,
        uint32_t level,
        uint32_t index,
        const string &folder
        ) {

    uint32_t count = octreeSparse[index].isParent ? itsSubsampleLUTs[index]->pointCount() : octreeSparse[index].pointCount;
    uint32_t validPoints = count;
    for (uint32_t u = 0; u < count; ++u)
    {
        if(octreeSparse[index].isParent) {
            auto lut = itsSubsampleLUTs[index]->toHost();
            if(lut[u] == INVALID_INDEX) {
                --validPoints;
            }
        }
        else {
            if(dataLUT[octreeSparse[index].chunkDataIndex + u] == INVALID_INDEX) {
                --validPoints;
            }
        }
    }

    if(octreeSparse[index].isFinished && validPoints > 0) {
        string name = octreeSparse[index].isParent ? (folder + "//_parent_") : (folder + "//_leaf_");
        std::ofstream ply;
        ply.open (name + std::to_string(level) +
                  "_" +
                  std::to_string(index) +
                  "_" +
                  std::to_string(validPoints) +
                  ".ply",
                  std::ios::binary
        );

        ply << "ply\n"
               "format binary_little_endian 1.0\n"
               "comment Created by AIT Austrian Institute of Technology\n"
               "element vertex "
            << validPoints
            << "\n"
               "property float x\n"
               "property float y\n"
               "property float z\n"
               "end_header\n";
        for (uint32_t u = 0; u < count; ++u)
        {
            if(octreeSparse[index].isParent) {
                auto lut = itsSubsampleLUTs[index]->toHost();
                if(lut[u] != INVALID_INDEX) {
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[lut[u]].x)), sizeof (float));
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[lut[u]].y)), sizeof (float));
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[lut[u]].z)), sizeof (float));
                }
            }
            else {
                if(dataLUT[octreeSparse[index].chunkDataIndex + u] != INVALID_INDEX) {
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[dataLUT[octreeSparse[index].chunkDataIndex + u]].x)), sizeof (float));
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[dataLUT[octreeSparse[index].chunkDataIndex + u]].y)), sizeof (float));
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[dataLUT[octreeSparse[index].chunkDataIndex + u]].z)), sizeof (float));
                }
            }
        }
        ply.close ();
    }
    if (level > 0) {
        for(int i = 0; i < octreeSparse[index].childrenChunksCount; ++i) {
            count += exportTreeNode(cpuPointCloud, octreeSparse, dataLUT, level - 1, octreeSparse[index].childrenChunks[i], folder);
        }
    }
    return count;
}

void SparseOctree::exportOctree(const string &folderPath) {
    auto cpuPointCloud = itsCloudData->toHost();
    auto octreeSparse = itsOctreeSparse->toHost();
    auto dataLUT = itsDataLUT->toHost();
    uint32_t topLevelIndex = itsVoxelAmountSparse->toHost()[0] - 1;

    // ToDo: Remove .get() -> pass unique_ptr by reference
    uint32_t exportedPoints = exportTreeNode(cpuPointCloud.get(), octreeSparse, dataLUT, itsGlobalOctreeDepth, topLevelIndex, folderPath);
    assert(exportedPoints == itsMetadata.pointAmount);
    spdlog::info("Sparse octree ({}/{} points) exported to: {}", exportedPoints, itsMetadata.pointAmount, folderPath);
    spdlog::info("{}", itsSubsampleLUTs.size());
}

void SparseOctree::freeGpuMemory() {
    itsDensePointCountPerVoxel.reset();
    itsDenseToSparseLUT.reset();
    itsSparseToDenseLUT.reset();
    itsVoxelAmountSparse.reset();
    itsOctreeSparse.reset();
    spdlog::debug("Sparse octree GPU memory deleted");
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

        itsTimeMeasurement.insert(std::make_pair("initializeOctreeSparse_" + std::to_string(itsGridSideLengthPerLevel[i]), time));
    }

    spdlog::info("'initializeOctreeSparse' took {:f} [ms]", timeAccumulated);
}