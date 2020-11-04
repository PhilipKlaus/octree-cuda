//
// Created by KlausP on 02.11.2020.
//

#ifndef OCTREE_LIBRARY_OCTREEBASE_H
#define OCTREE_LIBRARY_OCTREEBASE_H

#include <cstdint>
#include <map>

#include <cudaArray.h>
#include "types.h"

using namespace std;

class OctreeBase {

public:
    explicit OctreeBase(PointCloudMetadata cloudMetadata, unique_ptr<CudaArray<Vector3>> cloudData) :
    itsMetadata(cloudMetadata),
    itsCloudData(move(cloudData))
    {
        itsDataLUT = make_unique<CudaArray<uint32_t>>(cloudMetadata.pointAmount, "Data LUT");
    }

    virtual void initialPointCounting(uint32_t initialDepth) = 0;
    virtual void performCellMerging(uint32_t threshold) = 0;
    virtual void distributePoints() = 0;
    virtual void exportOctree(const string &folderPath) = 0;

public:
    OctreeBase(const OctreeBase&) = delete;
    void operator=(const OctreeBase&) = delete;

protected:
    PointCloudMetadata itsMetadata;
    unique_ptr<CudaArray<Vector3>> itsCloudData;
    unique_ptr<CudaArray<uint32_t>> itsDataLUT;         // LUT for accessing point cloud data from the octree
    map<std::string, float> itsTimeMeasurement;

};

#endif //OCTREE_LIBRARY_OCTREEBASE_H
