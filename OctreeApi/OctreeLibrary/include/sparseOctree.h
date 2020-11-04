//
// Created by KlausP on 02.11.2020.
//

#ifndef OCTREE_LIBRARY_SPARSE_OCTREE_H
#define OCTREE_LIBRARY_SPARSE_OCTREE_H

#include "octreeBase.h"

class SparseOctree : public OctreeBase {

public:
    SparseOctree(PointCloudMetadata cloudMetadata, unique_ptr<CudaArray<Vector3>> cloudData) :
            OctreeBase(cloudMetadata, std::move(cloudData)),
            itsGlobalOctreeDepth(0),
            itsGlobalOctreeBase(0),
            itsVoxelAmountDense(0)
    {
        spdlog::info("Instantiated sparse octree for {} points", cloudMetadata.pointAmount);
    }

    void initialPointCounting(uint32_t initialDepth) override;
    void performCellMerging(uint32_t threshold) override;
    void distributePoints() override;
    void exportOctree(const string &folderPath) override;
    void freeGpuMemory() override;

    // Debugging methods
    unique_ptr<uint32_t[]> getDensePointCountPerVoxel();
    unique_ptr<int[]> getDenseToSparseLUT();
    unique_ptr<Chunk[]> getOctreeSparse();
    uint32_t getVoxelAmountSparse();

private:
    void initializeOctreeSparse(uint32_t threshold);
    void initializeBaseGridSparse();
    uint32_t exportTreeNode(Vector3* cpuPointCloud, const unique_ptr<Chunk[]> &octreeSparse, const unique_ptr<uint32_t[]> &dataLUT, uint32_t level, uint32_t index, const string &folder);

private:
    uint32_t itsGlobalOctreeDepth;
    uint32_t itsGlobalOctreeBase;
    uint32_t itsVoxelAmountDense;

    unique_ptr<CudaArray<uint32_t>> itsDensePointCountPerVoxel;     // Holds all point counts in dense form
    unique_ptr<CudaArray<int>> itsDenseToSparseLUT;                 // LUT for mapping from dense to sparse
    unique_ptr<CudaArray<uint32_t>> itsVoxelAmountSparse;    // Overall initial cell amount of the sparse octree
    unique_ptr<CudaArray<Chunk>> itsOctreeSparse;       // Holds the sparse octree

};

#endif //OCTREE_LIBRARY_SPARSE_OCTREE_H
