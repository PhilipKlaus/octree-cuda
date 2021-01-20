//
// Created by KlausP on 01.11.2020.
//

#ifndef OCTREE_API_SESSION_H
#define OCTREE_API_SESSION_H

#include <string>
#include <api_types.h>


class Session
{
public:
    Session (int device);
    ~Session ();

    // Avoid object copy
    Session(const Session&) = delete;
    void operator=(const Session&) = delete;

    static Session* ToSession (void* session);
    void setDevice() const;
    void setPointCloudHost(uint8_t *pointCloud);
    void setMetadata(const PointCloudMetadata &metadata);

    void generateOctree();
    void configureChunking(GridSize chunkingGrid, uint32_t mergingThreshold);
    void configureSubsampling(GridSize subsamplingGrid, SubsamplingStrategy strategy);
    void configureOctreeExport(const std::string &directory);
    void configureMemoryReport(const std::string &filename);
    void configureJsonReport(const std::string &filename);
    void configurePointDistributionReport(const std::string &filename, uint32_t);

private:
    template <typename coordinateType, typename colorType>
    void generateOctreeTemplated();

private:
    int itsDevice;
    PointCloudMetadata itsMetadata{};
    uint8_t *itsPointCloud;

    GridSize itsChunkingGrid = GRID_128;
    uint32_t itsMergingThreshold = 0;

    GridSize itsSubsamplingGrid = GRID_128;
    SubsamplingStrategy itsSubsamplingStrategy = RANDOM_POINT;

    std::string itsPointDistReportFile = "";
    std::string itsJsonReportFile = "";
    std::string itsOctreeExportDirectory = "";
    uint32_t itsPointDistributionBinWidth = 0;
};

#endif //OCTREE_API_SESSION_H
