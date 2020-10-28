//
// Created by KlausP on 05.10.2020.
//

#ifndef POINT_CLOUD_H
#define POINT_CLOUD_H

#include "types.h"
#include "cudaArray.h"


class PointCloud {

public:
    explicit PointCloud(std::unique_ptr<CudaArray<Vector3>> data):
            itsCloudData(move(data)),
            itsCellAmount(0),
            itsGridBaseSideLength(0),
            itsMetadata({})
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

    PointCloudMetadata& getMetadata() { return itsMetadata; }
    unique_ptr<Chunk[]> getOctree();
    unique_ptr<uint32_t[]> getDataLUT();


private:

    uint32_t exportTreeNode(Vector3* cpuPointCloud, const unique_ptr<Chunk[]> &octree, const unique_ptr<uint32_t[]> &dataLUT, uint32_t level, uint32_t index);


private:
    // Point cloud
    unique_ptr<CudaArray<Vector3>> itsCloudData;    // The point cloud data
    PointCloudMetadata itsMetadata;                 // Point cloud metadata

    // Octree
    unique_ptr<CudaArray<Chunk>> itsOctree;         // The actual hierarchical octree data structure
    unique_ptr<CudaArray<uint32_t>> itsDataLUT;     // LUT for accessing point cloud data from the octree

    // Octree Metadata
    uint32_t itsCellAmount;                         // Overall initial cell amount of the octree
    uint32_t itsGridBaseSideLength;                 // The side length of the lowest grid in the octree (e.g. 128)

    // Time measurements
    float itsInitialPointCountTime;
    std::vector<float> itsMergingTime;
    float itsDistributionTime;
};

#endif //POINT_CLOUD_H
