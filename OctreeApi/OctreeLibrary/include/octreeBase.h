//
// Created by KlausP on 02.11.2020.
//

#ifndef OCTREE_LIBRARY_OCTREEBASE_H
#define OCTREE_LIBRARY_OCTREEBASE_H

#include <cstdint>
#include <map>

#include <cudaArray.h>
#include <types.h>
#include <tools.cuh>

using namespace std;

class OctreeBase {

public:
    explicit OctreeBase(PointCloudMetadata cloudMetadata, unique_ptr<CudaArray<Vector3>> cloudData) :
            itsMetadata(cloudMetadata),
            itsCloudData(move(cloudData)),
            itsGlobalOctreeDepth(0),
            itsGobalOctreeSideLength(0),
            itsVoxelAmountDense(0)
    {
        itsDataLUT = make_unique<CudaArray<uint32_t>>(cloudMetadata.pointAmount, "Data LUT");
    }

public:
    virtual void initialPointCounting(uint32_t initialDepth) = 0;
    virtual void performCellMerging(uint32_t threshold) = 0;
    virtual void distributePoints() = 0;
    virtual void performIndexing() = 0;
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

    void preCalculateOctreeParameters(uint32_t octreeDepth) {
        itsGlobalOctreeDepth = octreeDepth;

        // Precalculate parameters
        itsGobalOctreeSideLength = static_cast<uint32_t >(pow(2, octreeDepth));
        for(uint32_t gridSize = itsGobalOctreeSideLength; gridSize > 0; gridSize >>= 1) {
            itsGridSideLengthPerLevel.push_back(gridSize);
            itsLinearizedDenseVoxelOffset.push_back(itsVoxelAmountDense);
            itsVoxelsPerLevel.push_back(static_cast<uint32_t>(pow(gridSize, 3)));
            itsVoxelAmountDense += static_cast<uint32_t>(pow(gridSize, 3));
        }
    }

    void calculateVoxelBB(BoundingBox &bb, Vector3i &coords, uint32_t denseVoxelIndex, uint32_t level) {

        // 1. Calculate coordinates of voxel within the actual level
        auto indexInVoxel = denseVoxelIndex - itsLinearizedDenseVoxelOffset[level];
        tools::mapFromDenseIdxToDenseCoordinates(coords, indexInVoxel, itsGridSideLengthPerLevel[level]);

        // 2. Calculate the bounding box for the actual voxel
        // ToDo: Include scale and offset!!!
        auto dimension = tools::subtract(itsMetadata.boundingBox.maximum, itsMetadata.boundingBox.minimum);
        auto width = dimension.x / itsGridSideLengthPerLevel[level];
        auto height = dimension.y / itsGridSideLengthPerLevel[level];
        auto depth = dimension.z / itsGridSideLengthPerLevel[level];

        bb.minimum.x = coords.x * width;
        bb.minimum.y = coords.y * height;
        bb.minimum.z = coords.z * depth;
        bb.maximum.x = (coords.x + 1.f) * width;
        bb.maximum.y = (coords.y + 1.f) * height;
        bb.maximum.z = (coords.z + 1.f) * depth;
    }

public:
    OctreeBase(const OctreeBase&) = delete;
    void operator=(const OctreeBase&) = delete;

protected:
    PointCloudMetadata itsMetadata;                             // The metadata associated with the cloud
    unique_ptr<CudaArray<Vector3>> itsCloudData;                // The cloud data
    unique_ptr<CudaArray<uint32_t>> itsDataLUT;                 // LUT for accessing point cloud data from the octree
    unordered_map<std::string, float> itsTimeMeasurement;       // Holds all time measurements in the form (measurementName, time)

    uint32_t itsGlobalOctreeDepth;                              // The depth of the global octree
    uint32_t itsVoxelAmountDense;                               // The amount of dense voxels within the octree hierarchy
    uint32_t itsGobalOctreeSideLength;                          // The side-length of the octree's base

    // Pre-calculations
    vector<uint32_t> itsVoxelsPerLevel ;                        // Holds the voxel amount per level (dense)
    vector<uint32_t> itsGridSideLengthPerLevel;                 // Holds the side length of the grid per level
                                                                // E.g.: level 0 -> 128x128x128 -> side length: 128
    vector<uint32_t> itsLinearizedDenseVoxelOffset;             // Holds the linear voxel offset for each level (dense)
                                                                // Level 0 is e.g. 128x128x128
                                                                // Offset for level 0 = 0
                                                                // Offset for level 1 = level 0 + 128 x 128 x128
};

#endif //OCTREE_LIBRARY_OCTREEBASE_H
