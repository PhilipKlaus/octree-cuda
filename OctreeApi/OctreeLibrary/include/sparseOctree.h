//
// Created by KlausP on 02.11.2020.
//

#ifndef OCTREE_LIBRARY_SPARSE_OCTREE_H
#define OCTREE_LIBRARY_SPARSE_OCTREE_H

#include "octreeBase.h"

class SparseOctree : public OctreeBase {

public:
    SparseOctree(PointCloudMetadata cloudMetadata, unique_ptr<CudaArray<Vector3>> cloudData) :
            OctreeBase(cloudMetadata, std::move(cloudData))
    {
        spdlog::info("Instantiated sparse octree for {} points", cloudMetadata.pointAmount);
    }

    void initialPointCounting(uint32_t initialDepth) override;
    void performCellMerging(uint32_t threshold) override;
    void distributePoints() override;
    void performIndexing() override;

    void exportOctree(const string &folderPath) override;
    void freeGpuMemory() override;

    // Debugging methods
    unique_ptr<uint32_t[]> getDensePointCountPerVoxel();
    unique_ptr<int[]> getDenseToSparseLUT();
    unique_ptr<int[]> getSparseToDenseLUT();
    unique_ptr<Chunk[]> getOctreeSparse();
    uint32_t getVoxelAmountSparse();
    unordered_map<uint32_t, unique_ptr<CudaArray<uint32_t>>> const& getSubsampleLUT() const;

private:

    // Merging
    void initializeOctreeSparse(uint32_t threshold);
    void initializeBaseGridSparse();

    // Indexing
    void evaluateOctreeStatistics(const unique_ptr<Chunk[]> &h_octreeSparse, uint32_t sparseVoxelIndex);
    void hierarchicalCount(const unique_ptr<Chunk[]> &h_octreeSparse,
                           const unique_ptr<int[]> &h_sparseToDenseLUT,
                           uint32_t sparseVoxelIndex,
                           uint32_t level,
                           unique_ptr<CudaArray<uint32_t>> &subsampleCountingGrid,
                           unique_ptr<CudaArray<int>> &subsampleDenseToSparseLUT,
                           unique_ptr<CudaArray<uint32_t>> &subsampleSparseVoxelCount);

    // Exporting
    uint32_t exportTreeNode(Vector3* cpuPointCloud, const unique_ptr<Chunk[]> &octreeSparse, const unique_ptr<uint32_t[]> &dataLUT, uint32_t level, uint32_t index, const string &folder);

private:

    unique_ptr<CudaArray<uint32_t>> itsDensePointCountPerVoxel;                 // Holds all point counts in dense form
    unique_ptr<CudaArray<int>> itsDenseToSparseLUT;                             // LUT for mapping from dense to sparse
    unique_ptr<CudaArray<int>> itsSparseToDenseLUT;                             // LUT for mapping from sparse to dense
    unique_ptr<CudaArray<uint32_t>> itsVoxelAmountSparse;                       // Overall initial cell amount of the sparse octree
    unique_ptr<CudaArray<Chunk>> itsOctreeSparse;                               // Holds the sparse octree

    unordered_map<uint32_t, unique_ptr<CudaArray<uint32_t>>> itsSubsampleLUTs;
};

#endif //OCTREE_LIBRARY_SPARSE_OCTREE_H
