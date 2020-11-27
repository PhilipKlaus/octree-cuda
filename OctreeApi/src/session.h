//
// Created by KlausP on 01.11.2020.
//

#ifndef OCTREE_API_SESSION_H
#define OCTREE_API_SESSION_H

#include <sparseOctree.h>

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
    void setOctreeProperties(uint16_t globalOctreeLevel, uint32_t mergingThreshold);
    void generateOctree();
    void exportPlyNodes(const string &filename);
    void configureMemoryReport(const std::string &filename);
    void exportTimeMeasurements(const std::string &filename);
    void exportOctreeStatistics(const std::string &filename);
    void configurePointDistributionReport(const std::string &filename, uint32_t);

private:
    int itsDevice;
    PointCloudMetadata itsMetadata{};
    unique_ptr<SparseOctree> itsOctree;
    unique_ptr<CudaArray<uint8_t>> data;

    // Octree Configuration
    uint16_t itsGlobalOctreeLevel{};
    uint32_t itsMergingThreshold{};

    string itsPointDistributionReport = "";
    uint32_t itsPointDistributionBinWidth = 0;
};

#endif //OCTREE_API_SESSION_H
