//
// Created by KlausP on 05.10.2020.
//

#ifndef POINT_CLOUD_H
#define POINT_CLOUD_H

#include "cudaArray.h"
#include "octreeApi.h"

#include <memory>

using namespace std;


class PointCloud {

public:
    explicit PointCloud(std::unique_ptr<CudaArray<Vector3>> data):
            itsCloudData(move(data)),
            itsCellAmount(0),
            itsGridBaseSideLength(0),
            itsMetadata({}),
            itsOctreeLevels(0),
            itsMergingThreshold(0)
    {
        itsDataLUT = make_unique<CudaArray<uint32_t>>(itsCloudData->pointCount(), "Data LUT");
    }

    // Avoid object copy
    PointCloud(const PointCloud&) = delete;
    void operator=(const PointCloud&) = delete;

    // Pipeline
    void initialPointCounting(uint32_t initialDepth);
    void initialPointCountingSparse(uint32_t initialDepth);
    void performCellMerging(uint32_t threshold);
    void performCellMergingSparse(uint32_t threshold);
    void distributePoints();
    void distributePointsSparse();

    // Export functions
    void exportOctree(Vector3* cpuPointCloud);
    void exportTimeMeasurement();
    void exportOctreeSparse(Vector3 *cpuPointCloud);

    PointCloudMetadata& getMetadata() { return itsMetadata; }
    unique_ptr<Chunk[]> getOctree();
    unique_ptr<uint32_t[]> getDataLUT();

    // Sparse Octree functions
    unique_ptr<Chunk[]> getOctreeSparse();
    unique_ptr<uint32_t[]> getDensePointCount();
    unique_ptr<int[]> getDenseToSparseLUT();
    uint32_t getCellAmountSparse();

private:

    uint32_t exportTreeNode(Vector3* cpuPointCloud, const unique_ptr<Chunk[]> &octree, const unique_ptr<uint32_t[]> &dataLUT, uint32_t level, uint32_t index);
    uint32_t exportTreeNodeSparse(Vector3* cpuPointCloud, const unique_ptr<Chunk[]> &octreeSparse, const unique_ptr<uint32_t[]> &dataLUT, uint32_t level, uint32_t index);

    void initializeOctreeSparse();
    void initializeBaseGridSparse();

private:
    // Point cloud
    unique_ptr<CudaArray<Vector3>> itsCloudData;    // The point cloud data
    PointCloudMetadata itsMetadata;                 // Point cloud metadata

    // Octree
    unique_ptr<CudaArray<Chunk>> itsOctree;         // The actual hierarchical octree data structure
    unique_ptr<CudaArray<uint32_t>> itsDataLUT;     // LUT for accessing point cloud data from the octree

    // Dense Octree Metadata
    uint32_t itsCellAmount;                         // Overall initial cell amount of the octree
    uint32_t itsGridBaseSideLength;                 // The side length of the lowest grid in the octree (e.g. 128)

    // Sparse Octree
    unique_ptr<CudaArray<Chunk>> itsOctreeSparse;       // Holds the sparse octree
    unique_ptr<CudaArray<uint32_t>> itsDensePointCount; // Holds all point counts in dense form
    unique_ptr<CudaArray<int>> itsDenseToSparseLUT;     // LUT for mapping from dense to sparse

    // Sparse Octree Metadata
    unique_ptr<CudaArray<uint32_t>> itsCellAmountSparse;    // Overall initial cell amount of the sparse octree
    uint32_t itsOctreeLevels;
    uint32_t itsMergingThreshold;


    // Time measurements
    float itsInitialPointCountTime;
    std::vector<float> itsMergingTime;
    float itsDistributionTime;
};

#endif //POINT_CLOUD_H
