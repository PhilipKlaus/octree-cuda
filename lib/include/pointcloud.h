//
// Created by KlausP on 05.10.2020.
//

#ifndef OCTREECUDA_POINTCLOUD_H
#define OCTREECUDA_POINTCLOUD_H

#include "types.h"
#include <vector>

class PointCloud {

public:
    explicit PointCloud(unique_ptr<CudaArray<Vector3>> data)
    : itsData(move(data)), itsCellAmount(0) {
        itsTreeData = make_unique<CudaArray<uint64_t>>(itsData->pointCount(), "LUT");
    }

    void initialPointCounting(uint64_t initialDepth);
    void performCellMerging(uint64_t threshold);
    void exportToPly(const std::string& file_name);
    void distributePoints();
    void exportGlobalTree();
    PointCloudMetadata& getMetadata() { return itsMetadata; }

    unique_ptr<Chunk[]> getCountingGrid();
    unique_ptr<uint64_t[]> getTreeData();

private:
    // Point cloud data
    unique_ptr<CudaArray<Vector3>> itsData;
    PointCloudMetadata itsMetadata;

    // Grid / Octree
    uint64_t itsCellAmount; // Overall cell amount of the complete hierarchy pyramide
    uint64_t itsGridBaseSideLength;
    unique_ptr<CudaArray<Chunk>> itsGrid;

    unique_ptr<CudaArray<uint64_t>> itsTreeData;
};

#endif //OCTREECUDA_POINTCLOUD_H
