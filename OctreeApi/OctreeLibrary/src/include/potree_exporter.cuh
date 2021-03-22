#pragma once

#include "octree_exporter.cuh"


class PotreeExporter : public OctreeExporter
{
public:
    PotreeExporter (
            const PointCloud& pointCloud,
            const std::shared_ptr<Chunk[]>& octree,
            OctreeMetadata metadata,
            PointCloudMetadata cloudMetadata,
            SubsampleMetadata subsamplingMetadata);

    void exportOctree (const std::string& path) override;

private:
    void createBinaryHierarchyFiles ();
    void breathFirstExport (std::ofstream& pointFile, std::ofstream& hierarchyFile);
    inline uint8_t getChildMask (uint32_t nodeIndex);
    void createMetadataFile ();

private:
    std::string itsExportFolder;
    uint32_t itsExportedNodes;

#pragma pack(push, 1)
    struct HierarchyFileEntry
    {
        uint8_t type;
        uint8_t bitmask;
        uint32_t points;
        uint64_t byteOffset;
        uint64_t byteSize;
    };
#pragma pack(pop)
};
