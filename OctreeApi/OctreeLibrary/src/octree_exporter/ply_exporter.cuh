#include "octree_exporter.cuh"


template <typename coordinateType, typename colorType>
class PlyExporter : public OctreeExporter<coordinateType, colorType>
{
public:
    PlyExporter (
            const PointCloud& pointCloud,
            const std::shared_ptr<Chunk[]>& octree,
            const GpuArrayU32& leafeLut,
            const unordered_map<uint32_t, GpuArrayU32>& parentLut,
            const unordered_map<uint32_t, GpuAveraging>& parentAveraging,
            OctreeMetadata metadata,
            SubsampleMetadata subsamplingMetadata);

    void exportOctree (const std::string& path) override;

private:
    void exportNode (uint32_t nodeIndex, const string& octreeLevel, const std::string& path);
    void createPlyHeader (string& header, uint32_t pointsToExport);
    void writePointCoordinates (
            const std::unique_ptr<uint8_t[]>& buffer, uint64_t bufferOffset, uint64_t pointByteIndex);
    void writeColorAveraged (
            const std::unique_ptr<uint8_t[]>& buffer, uint64_t bufferOffset, uint32_t nodeIndex, uint32_t pointIndex);
    void writeColorNonAveraged (
            const std::unique_ptr<uint8_t[]>& buffer, uint64_t bufferOffset, uint64_t pointByteIndex);
};
