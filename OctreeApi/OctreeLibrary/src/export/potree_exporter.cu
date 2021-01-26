#include "potree_exporter.cuh"
#include <json.hpp>


template <typename coordinateType, typename colorType>
PotreeExporter<coordinateType, colorType>::PotreeExporter (
        const GpuArrayU8& pointCloud,
        const GpuOctree& octree,
        const GpuArrayU32& leafeLut,
        const unordered_map<uint32_t, GpuArrayU32>& parentLut,
        const unordered_map<uint32_t, GpuAveraging>& parentAveraging,
        OctreeMetadata<coordinateType> metadata) :
        OctreeExporter<coordinateType, colorType> (pointCloud, octree, leafeLut, parentLut, parentAveraging, metadata)
{}

template <typename coordinateType, typename colorType>
void PotreeExporter<coordinateType, colorType>::exportOctree (const std::string& path)
{
    createBinaryFile();
    createHierarchyFile();
    createMetadataFile();
}

template <typename coordinateType, typename colorType>
void PotreeExporter<coordinateType, colorType>::createBinaryFile ()
{
    traverseNode(this->getRootIndex());
}

template <typename coordinateType, typename colorType>
void PotreeExporter<coordinateType, colorType>::createHierarchyFile ()
{}

template <typename coordinateType, typename colorType>
void PotreeExporter<coordinateType, colorType>::createMetadataFile ()
{
    spdlog::info("TEST: {}", this->itsPointsExported);
}


template <typename coordinateType, typename colorType>
void PotreeExporter<coordinateType, colorType>::traverseNode (uint32_t nodeIndex)
{
    bool isParent   = this->isParentNode(nodeIndex);
    bool isFinished = this->isFinishedNode(nodeIndex);

    auto pointsInNode = this->getPointsInNode(nodeIndex);
    const std::unique_ptr<uint32_t[]>& lut = isParent ? this->itsParentLut[nodeIndex] : this->itsLeafeLut;

    if (isFinished)
    {
        uint32_t validPoints  = 0;

        for (uint32_t u = 0; u < pointsInNode; ++u)
        {
            if (isParent)
            {
                if (lut[u] != INVALID_INDEX)
                {
                    ++validPoints;
                }
            }
            else
            {
                if (lut[this->itsOctree[nodeIndex].chunkDataIndex + u] != INVALID_INDEX)
                {
                    ++validPoints;
                }
            }
        }
        this->itsPointsExported += validPoints;
    }

    for (auto i = 0; i < 8; ++i)
    {
        int childNodeIndex = this->getChildNodeIndex(nodeIndex, i);
        if (childNodeIndex != -1)
        {
            traverseNode (childNodeIndex);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
//                                           SparseOctree<float, uint8_t>
//----------------------------------------------------------------------------------------------------------------------
template PotreeExporter<float, uint8_t>::PotreeExporter (
        const GpuArrayU8& pointCloud,
        const GpuOctree& octree,
        const GpuArrayU32& leafeLut,
        const unordered_map<uint32_t, GpuArrayU32>& parentLut,
        const unordered_map<uint32_t, GpuAveraging>& parentAveraging,
        OctreeMetadata<float> metadata);

template void PotreeExporter<float, uint8_t>::exportOctree (const std::string& path);

//----------------------------------------------------------------------------------------------------------------------
//                                           SparseOctree<double, uint8_t>
//----------------------------------------------------------------------------------------------------------------------
template PotreeExporter<double, uint8_t>::PotreeExporter (
        const GpuArrayU8& pointCloud,
        const GpuOctree& octree,
        const GpuArrayU32& leafeLut,
        const unordered_map<uint32_t, GpuArrayU32>& parentLut,
        const unordered_map<uint32_t, GpuAveraging>& parentAveraging,
        OctreeMetadata<double> metadata);

template void PotreeExporter<double, uint8_t>::exportOctree (const std::string& path);