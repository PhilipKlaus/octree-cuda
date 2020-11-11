//
// Created by KlausP on 04.11.2020.
//

#ifndef OCTREE_LIBRARY_DENSE_OCTREE_H
#define OCTREE_LIBRARY_DENSE_OCTREE_H


#include "octreeBase.h"

class DenseOctree : public OctreeBase {

public:
    DenseOctree(PointCloudMetadata cloudMetadata, unique_ptr<CudaArray<Vector3>> cloudData) :
            OctreeBase(cloudMetadata, std::move(cloudData)),
            itsGlobalOctreeDepth(0),
            itsGlobalOctreeBase(0),
            itsVoxelAmountDense(0)
    {
        spdlog::info("Instantiated dense octree for {} points", cloudMetadata.pointAmount);
    }

    void initialPointCounting(uint32_t initialDepth) override;
    void performCellMerging(uint32_t threshold) override;
    void distributePoints() override;
    void performIndexing() override {}
    void exportOctree(const string &folderPath) override;
    void freeGpuMemory() override;

    // Debugging methods
    //unique_ptr<uint32_t[]> getDensePointCountPerVoxel();
    //unique_ptr<int[]> getDenseToSparseLUT();
    unique_ptr<Chunk[]> getOctreeDense();
    //uint32_t getVoxelAmountSparse();

private:
    uint32_t exportTreeNode(Vector3* cpuPointCloud, const unique_ptr<Chunk[]> &octree, const unique_ptr<uint32_t[]> &dataLUT, uint32_t level, uint32_t index, const string &folderPath);

private:
    uint32_t itsGlobalOctreeDepth;
    uint32_t itsGlobalOctreeBase;
    uint32_t itsVoxelAmountDense;

    unique_ptr<CudaArray<Chunk>> itsOctreeDense;       // Holds the dense octree

};


#endif //OCTREE_LIBRARY_DENSE_OCTREE_H
