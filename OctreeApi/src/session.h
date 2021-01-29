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

    void setCloudType (CloudType cloudType);
    void setCloudPointAmount (uint32_t pointAmount);
    void setCloudDataStride (uint32_t dataStride);
    void setCloudScaleF (float x, float y, float z);
    void setCloudOffsetF (float x, float y, float z);
    void setCloudBoundingBoxF (float minX, float minY, float minZ, float maxX, float maxY, float maxZ);
    void setCloudScaleD (double x, double y, double z);
    void setCloudOffsetD (double x, double y, double z);
    void setCloudBoundingBoxD (double minX, double minY, double minZ, double maxX, double maxY, double maxZ);

    void generateOctree ();
    void configureChunking (GridSize chunkingGrid, uint32_t mergingThreshold);
    void configureSubsampling (GridSize subsamplingGrid, SubsamplingStrategy strategy);
    void configureOctreeExport (const std::string& directory);
    void configureMemoryReport (const std::string& filename);
    void configureJsonReport (const std::string& filename);
    void configurePointDistributionReport (const std::string& filename, uint32_t);

private:
    template <typename coordinateType, typename colorType>
    void generateOctreeTemplated (PointCloudMetadata<coordinateType> metadata);

private:
    int itsDevice;
    uint8_t* itsPointCloud;

    // Cloud metadata
    CloudType itsCloudType              = CloudType::CLOUD_FLOAT_UINT8_T;
    uint32_t itsPointAmount             = 0;
    uint32_t itsDataStride              = 0;
    Vector3<float> itsScaleF            = {};
    Vector3<double> itsScaleD           = {};
    Vector3<float> itsOffsetF           = {};
    Vector3<double> itsOffsetD          = {};
    BoundingBox<float> itsBoundingBoxF  = {};
    BoundingBox<double> itsBoundingBoxD = {};

    GridSize itsChunkingGrid     = GRID_128;
    uint32_t itsMergingThreshold = 0;

    GridSize itsSubsamplingGrid                = GRID_128;
    SubsamplingStrategy itsSubsamplingStrategy = RANDOM_POINT;

    std::string itsPointDistReportFile    = "";
    std::string itsJsonReportFile         = "";
    std::string itsOctreeExportDirectory  = "";
    uint32_t itsPointDistributionBinWidth = 0;
};

#endif // OCTREE_API_SESSION_H
