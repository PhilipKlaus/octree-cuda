//
// Created by KlausP on 05.10.2020.
//

#ifndef POINT_CLOUD_H
#define POINT_CLOUD_H


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

    // Avoid object copy
    PointCloud(const PointCloud&) = delete;
    void operator=(const PointCloud&) = delete;

    void initialPointCounting(uint32_t initialDepth);
    void performCellMerging(uint32_t threshold);
    void distributePoints();
    void exportOctree();

    PointCloudMetadata& getMetadata() { return itsMetadata; }
    unique_ptr<Chunk[]> getOctree();
    unique_ptr<Vector3[]> getChunkData();


private:

    uint32_t exportTreeNode(const unique_ptr<Chunk[]> &octree, const unique_ptr<Vector3[]> &chunkData, uint32_t level, uint32_t index);


private:
    // Point cloud
    unique_ptr<CudaArray<Vector3>> itsCloudData;    // The point cloud data
    PointCloudMetadata itsMetadata;                 // Point cloud metadata

    // Octree
    unique_ptr<CudaArray<Chunk>> itsOctree;         // The actual hierarchical octree data structure
    unique_ptr<CudaArray<Vector3>> itsChunkData;    // Holding actual point cloud data for the octree
    uint32_t itsCellAmount;                         // Overall initial cell amount of the octree
    uint32_t itsGridBaseSideLength;                 // The side length of the lowest grid in the octree (e.g. 128)

};

#endif //POINT_CLOUD_H
