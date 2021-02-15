//
// Created by KlausP on 01.11.2020.
//

#ifndef OCTREE_API_SESSION_H
#define OCTREE_API_SESSION_H

#include "octree_processor.h"
#include <api_types.h>
#include <memory>
#include <string>


class Session
{
public:
    Session (int device);
    ~Session ();

    // Avoid object copy
    Session (const Session&) = delete;
    void operator= (const Session&) = delete;

    static Session* ToSession (void* session);
    void setDevice () const;
    void setPointCloudHost (uint8_t* pointCloud);

    void setCloudType (uint8_t cloudType);
    void setCloudPointAmount (uint32_t pointAmount);
    void setCloudDataStride (uint32_t dataStride);
    void setCloudScale (double x, double y, double z);
    void setCloudOffset (double x, double y, double z);
    void setCloudBoundingBox (double minX, double minY, double minZ, double maxX, double maxY, double maxZ);

    void generateOctree ();
    void configureChunking (uint32_t chunkingGrid, uint32_t mergingThreshold);
    void configureSubsampling (uint32_t subsamplingGrid, bool averaging, bool replacementScheme);

    void exportPotree (const std::string& directory);
    void exportJsonReport (const std::string& filename);
    void exportMemoryReport (const std::string& filename);
    void exportDistributionHistogram (const std::string& filename, uint32_t);


private:
    int itsDevice;
    uint8_t* itsPointCloud;
    std::unique_ptr<OctreeProcessorPimpl> itsProcessor;

    uint32_t itsChunkingGrid     = 128;
    uint32_t itsMergingThreshold = 0;

    PointCloudMetadata itsCloudMetadata      = {};
    SubsampleMetadata itsSubsamplingMetadata = {};
};

#endif // OCTREE_API_SESSION_H
