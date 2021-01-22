#include "octree_exporter.cuh"


template <typename coordinateType, typename colorType>
class PlyExporter : public OctreeExporter<coordinateType, colorType>
{
public:
    PlyExporter (
            const GpuArrayU8& pointCloud,
            const GpuOctree& octree,
            const GpuArrayU32& leafeLut,
            const unordered_map<uint32_t, GpuArrayU32>& parentLut,
            const unordered_map<uint32_t, GpuAveraging>& parentAveraging,
            OctreeMetadata metadata);

    void exportOctree (const std::string& path) override;

private:
    void exportNode (uint32_t nodeIndex, const string& octreeLevel, const std::string& path);
    uint32_t getValidPointAmount(uint32_t nodeIndex, uint32_t pointAmount);
    void createPlyHeader(string &header, uint32_t pointsToExport);

private:
    uint32_t itsPointsExported;
};
