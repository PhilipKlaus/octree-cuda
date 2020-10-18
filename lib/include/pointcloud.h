//
// Created by KlausP on 05.10.2020.
//

#ifndef OCTREECUDA_POINTCLOUD_H
#define OCTREECUDA_POINTCLOUD_H

#include "types.h"
#include <vector>

class PointCloud {

public:
    explicit PointCloud(unique_ptr<CudaArray<Vector3>> data)//, uint8_t bytesPerCoordinate)
    : itsData(move(data))
      /*itsBytesPerCoordinate(bytesPerCoordinate)*/ {
        itsTreeData = make_unique<CudaArray<Vector3>>(itsData->pointCount());
    }
    void initialPointCounting(uint32_t initialDepth);
    void performCellMerging(uint32_t threshold);
    void exportToPly(const std::string& file_name);
    void distributePoints();
    void exportGlobalTree();
    PointCloudMetadata& getMetadata() { return itsMetadata; }

    vector<unique_ptr<Chunk[]>> getCountingGrid();
    unique_ptr<Vector3[]> getTreeData();

private:
    unique_ptr<CudaArray<Vector3>> itsData;
    PointCloudMetadata itsMetadata;

    uint8_t itsBytesPerCoordinate;

    uint32_t itsInitialDepth;
    uint32_t itsGridSize;
    unique_ptr<CudaArray<Vector3>> itsTreeData;
    std::vector<unique_ptr<CudaArray<Chunk>>> itsGrid;
    unique_ptr<CudaArray<uint32_t>> itsCounter;
    uint32_t itsThreshold;

};

#endif //OCTREECUDA_POINTCLOUD_H
