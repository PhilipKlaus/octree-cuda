/**
 * @file octree_processor.h
 * @author Philip Klaus
 * @brief Contains declarations for an OctreeProcessor using the Pimpl idiom
 */

#pragma once

#include "metadata.h"
#include <memory>
#include <string>
#include <vector>

class OctreeProcessor
{
public:
    OctreeProcessor (
            uint8_t* pointCloud,
            uint32_t chunkingGrid,
            uint32_t mergingThreshold,
            PointCloudMetadata cloudMetadata,
            SubsampleMetadata subsamplingMetadata);

    ~OctreeProcessor ();
    OctreeProcessor (const OctreeProcessor&) = delete;
    void operator= (const OctreeProcessor&) = delete;

    void initialPointCounting ();
    void performCellMerging ();
    void distributePoints ();
    void performSubsampling ();

    void exportPlyNodes (const std::string& folderPath);
    void exportPotree (const std::string& folderPath);
    void exportHistogram (const std::string& filePath, uint32_t binWidth);

    void updateStatistics ();
    const std::vector<std::tuple<std::string, float>>& getTimings ();
    const OctreeMetadata& getOctreeMetadata ();

private:
    class OctreeProcessorImpl;
    std::unique_ptr<OctreeProcessorImpl> itsProcessor;
};
