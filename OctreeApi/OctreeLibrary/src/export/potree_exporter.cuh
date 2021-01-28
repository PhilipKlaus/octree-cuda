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
    std::string itsExportFolder;
    void createBinaryHierarchyFiles ();
    uint64_t exportNode (uint32_t nodeIndex, uint64_t bytesWritten, std::ofstream& pointFile, std::ofstream& hierarchyFile);
    void breathFirstExport (std::ofstream& pointFile, std::ofstream& hierarchyFile);
    uint8_t writePointCoordinates (const std::unique_ptr<uint8_t[]>& buffer, uint64_t bufferOffset, uint64_t pointByteIndex);
    uint8_t writeColorAveraged (const std::unique_ptr<uint8_t[]>& buffer, uint64_t bufferOffset, uint32_t nodeIndex, uint32_t pointIndex);
    uint8_t writeColorNonAveraged (const std::unique_ptr<uint8_t[]>& buffer, uint64_t bufferOffset, uint64_t pointByteIndex);
    uint8_t getChildMask(uint32_t nodeIndex);

    void createMetadataFile ();
};
