#pragma once

#include "octree_exporter.cuh"
#include <future>
#include <thread>

struct ExportResult
{
    uint8_t type;
    uint8_t bitmask;
    uint32_t validPoints;
    uint64_t nodeByteSize;
    std::unique_ptr<uint8_t[]> buffer;
};


template <typename coordinateType, typename colorType>
class PotreeExporter : public OctreeExporter<coordinateType, colorType>
{
public:
    PotreeExporter (
            const PointCloud& pointCloud,
            const GpuOctree& octree,
            const GpuArrayU32& leafeLut,
            const unordered_map<uint32_t, GpuArrayU32>& parentLut,
            const unordered_map<uint32_t, GpuAveraging>& parentAveraging,
            OctreeMetadata metadata,
            SubsampleMetadata subsamplingMetadata);

    void exportOctree (const std::string& path) override;

private:
    void createBinaryHierarchyFiles ();
    ExportResult exportNode (uint32_t nodeIndex);
    void breathFirstExport (std::ofstream& pointFile, std::ofstream& hierarchyFile);
    inline uint8_t writeCoordinatesBuffered (
            const std::unique_ptr<uint8_t[]>& buffer, uint64_t bufferOffset, uint64_t pointByteIndex);
    inline uint8_t writeColorsBuffered (
            const std::unique_ptr<uint8_t[]>& buffer, uint64_t bufferOffset, uint32_t nodeIndex, uint32_t pointIndex);
    inline uint8_t writeSimpleColorsBuffered (
            const std::unique_ptr<uint8_t[]>& buffer, uint64_t bufferOffset, uint64_t pointByteIndex);
    inline uint8_t getChildMask (uint32_t nodeIndex);
    void createMetadataFile ();
    void exportBuffers (std::ofstream& pointFile, std::ofstream& hierarchyFile);

private:
    std::string itsExportFolder;
    std::vector<std::future<ExportResult>> itsFutureResults;
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
