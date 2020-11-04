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

public:
    virtual void initialPointCounting(uint32_t initialDepth) = 0;
    virtual void performCellMerging(uint32_t threshold) = 0;
    virtual void distributePoints() = 0;
    virtual void exportOctree(const string &folderPath) = 0;
    virtual void freeGpuMemory() = 0;

public:
    PointCloudMetadata& getMetadata() { return itsMetadata; }

    unique_ptr<uint32_t[]> getDataLUT() { return itsDataLUT->toHost(); }

    void exportTimeMeasurements(const string &filePath) {
        string headerLine, timeLine;
        ofstream timingCsv;
        timingCsv.open (filePath, ios::out);
        for (auto const& timeEntry : itsTimeMeasurement) {
            headerLine += (timeEntry.first + ",");
            timeLine += (to_string(timeEntry.second) + ",");
        }
        // Remove last colons
        headerLine = headerLine.substr(0, headerLine.size()-1);
        timeLine = timeLine.substr(0, timeLine.size()-1);
        timingCsv << headerLine << std::endl << timeLine;
        timingCsv.close();
        spdlog::info("Exported time measurements to {}", filePath);
    }

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
