//
// Created by KlausP on 05.10.2020.
//

#ifndef OCTREECUDA_POINTCLOUD_H
#define OCTREECUDA_POINTCLOUD_H

// Local dependencies
#include "types.h"


class PointCloud {

public:
    explicit PointCloud(unique_ptr<CudaArray<Vector3>> data):
            itsCloudData(move(data)),
            itsCellAmount(0),
            itsGridBaseSideLength(0),
            itsMetadata({})
    {
        itsChunkData = make_unique<CudaArray<Vector3>>(itsCloudData->pointCount(), "Chunk Data");
    }

    void initialPointCounting(uint64_t initialDepth);
    void performCellMerging(uint64_t threshold);
    void distributePoints();
    void exportGlobalTree();

    PointCloudMetadata& getMetadata() { return itsMetadata; }
    unique_ptr<Chunk[]> getCountingGrid();
    unique_ptr<Vector3[]> getTreeData();

private:
    // Data blocks
    unique_ptr<CudaArray<Vector3>> itsCloudData;    // The point cloud data
    unique_ptr<CudaArray<Vector3>> itsChunkData;    // The point cloud data grouped by chunk
    unique_ptr<CudaArray<Chunk>> itsGrid;           // The hierarchical grid structure

    PointCloudMetadata itsMetadata;                 // Cloud metadata

    // Grid / Octree
    uint64_t itsCellAmount;                         // Overall cell amount of the hierarchical grid
    uint64_t itsGridBaseSideLength;                 // The side length of the lowest grid in the hierarchy (e.g. 128)

};

#endif //OCTREECUDA_POINTCLOUD_H
