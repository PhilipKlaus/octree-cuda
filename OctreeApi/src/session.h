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
    void setOctreeProperties(GridSize chunkingGrid, GridSize subsamplingGrid, uint32_t mergingThreshold);
    void generateOctree();
    void configureOctreeExport(const string &directory);
    void configureMemoryReport(const std::string &filename);
    void configureJsonReport(const std::string &filename);
    void configurePointDistributionReport(const std::string &filename, uint32_t);

private:
    int itsDevice;
    PointCloudMetadata itsMetadata{};
    unique_ptr<SparseOctree> itsOctree;
    unique_ptr<CudaArray<uint8_t>> data;


    string itsPointDistReportFile = "";
    string itsJsonReportFile = "";
    string itsOctreeExportDirectory = "";
    uint32_t itsPointDistributionBinWidth = 0;
};

#endif //OCTREE_API_SESSION_H
