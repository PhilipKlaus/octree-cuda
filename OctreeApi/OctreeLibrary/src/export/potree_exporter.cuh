#pragma once

#include "octree_exporter.cuh"


template <typename coordinateType, typename colorType>
class PotreeExporter : public OctreeExporter<coordinateType, colorType>
{
public:
    PotreeExporter (
            const GpuArrayU8& pointCloud,
            const GpuOctree& octree,
            const GpuArrayU32& leafeLut,
            const unordered_map<uint32_t, GpuArrayU32>& parentLut,
            const unordered_map<uint32_t, GpuAveraging>& parentAveraging,
            OctreeMetadata<coordinateType> metadata);

    void exportOctree (const std::string& path) override;

private:
    void createBinaryFile();
    void createHierarchyFile();
    void createMetadataFile();
    void traverseNode (uint32_t nodeIndex);
};
