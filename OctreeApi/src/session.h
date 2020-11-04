//
// Created by KlausP on 01.11.2020.
//

#ifndef OCTREE_API_SESSION_H
#define OCTREE_API_SESSION_H

#include "../OctreeLibrary/include/pointcloud.h"
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
    const PointCloudMetadata& getMetadata() const;
    void generateOctree();
    void exportOctree(const string &filename);
    void configureMemoryReport(const std::string &filename);

private:
    int itsDevice;
    PointCloudMetadata itsMetadata{};
    unique_ptr<OctreeBase> itsOctree;
    unique_ptr<CudaArray<Vector3>> data;
    // Octree Configuration
    uint16_t itsGlobalOctreeLevel{};
    uint32_t itsMergingThreshold{};

};

#endif //OCTREE_API_SESSION_H
