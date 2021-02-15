//
// Created by KlausP on 02.11.2020.
//

#pragma once

#include "octree_metadata.h"
#include "api_types.h"
#include <memory>
#include <string>
#include <vector>

class OctreeProcessorPimpl
{
public:
    OctreeProcessorPimpl (
            uint8_t* pointCloud,
            uint32_t chunkingGrid,
            uint32_t mergingThreshold,
            PointCloudMetadata cloudMetadata,
            SubsampleMetadata subsamplingMetadata);
    ~OctreeProcessorPimpl ();

    OctreeProcessorPimpl (const OctreeProcessorPimpl&) = delete;
    void operator= (const OctreeProcessorPimpl&) = delete;

    void initialPointCounting ();
    void performCellMerging ();
    void distributePoints ();
    void performSubsampling ();
    void exportPlyNodes(const std::string& folderPath);
    void exportPotree(const std::string& folderPath);
    void exportHistogram(const std::string& filePath, uint32_t binWidth);
    void updateStatistics();
    const std::vector<std::tuple<std::string, float>>& getTimings ();
    const OctreeMetadata& getOctreeMetadata();

private:
    class OctreeProcessorImpl;
    std::unique_ptr<OctreeProcessorImpl> itsProcessor;
};
