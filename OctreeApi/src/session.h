//
// Created by KlausP on 01.11.2020.
//

#ifndef OCTREE_API_SESSION_H
#define OCTREE_API_SESSION_H

#include <api_types.h>
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
    void configureSubsampling (uint32_t subsamplingGrid, uint8_t strategy);
    void configureOctreeExport (const std::string& directory);
    void configureMemoryReport (const std::string& filename);
    void configureJsonReport (const std::string& filename);
    void configurePointDistributionReport (const std::string& filename, uint32_t);

private:
    template <typename coordinateType, typename colorType>
    void generateOctreeTemplated (PointCloudMetadata metadata);

private:
    int itsDevice;
    uint8_t* itsPointCloud;

    // Cloud metadata
    CloudType itsCloudType              = CloudType::CLOUD_FLOAT_UINT8_T;
    uint32_t itsPointAmount             = 0;
    uint32_t itsDataStride              = 0;
    Vector3<double> itsScale            = {};
    Vector3<double> itsOffset           = {};
    BoundingBox itsBoundingBox = {};

    // Chunking
    uint32_t itsChunkingGrid     = 128;
    uint32_t itsMergingThreshold = 0;

    // Subsampling
    uint32_t itsSubsamplingGrid                = 128;
    SubsamplingStrategy itsSubsamplingStrategy = RANDOM_POINT;

    std::string itsPointDistReportFile    = "";
    std::string itsJsonReportFile         = "";
    std::string itsOctreeExportDirectory  = "";
    uint32_t itsPointDistributionBinWidth = 0;
};

#endif // OCTREE_API_SESSION_H
