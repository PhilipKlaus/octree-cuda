#pragma once

#include "octree_exporter.cuh"


class PotreeExporter : public OctreeExporter
{
public:
    PotreeExporter ();

    void exportOctree (const std::string& path, const PointCloud& pointCloud,
                       const Octree& octree, const SubsampleMetadata& subsampleMetadata) override;

private:
    void createBinaryHierarchyFiles (const PointCloud& cloud, const Octree& octree);
    void breathFirstExport (std::ofstream& pointFile, std::ofstream& hierarchyFile, const Octree& octree);
    static inline uint8_t getChildMask (const Octree& octree, uint32_t nodeIndex);
    void createMetadataFile (const PointCloud& cloud, const SubsampleMetadata& subsampleMeta);

private:
    std::string itsExportFolder;
    uint32_t itsExportedNodes{};

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
