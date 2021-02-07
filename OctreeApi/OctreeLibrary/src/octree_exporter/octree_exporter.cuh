#pragma once

#include "octree_metadata.h"
#include "types.cuh"
#include "point_cloud.cuh"


template <typename coordinateType, typename colorType>
class OctreeExporter
{
public:
    OctreeExporter (
            const PointCloud& pointCloud,
            const GpuOctree& octree,
            const GpuArrayU32& leafeLut,
            const unordered_map<uint32_t, GpuArrayU32>& parentLut,
            const unordered_map<uint32_t, GpuAveraging>& parentAveraging,
            OctreeMetadata metadata) :
            itsMetadata (metadata),
            itsCloud (pointCloud->getCloudHost()), itsOctree (octree->toHost ()), itsLeafeLut (leafeLut->toHost ()),
            itsAbsorbedNodes (0), itsPointsExported (0)
    {
        std::for_each (parentLut.cbegin (), parentLut.cend (), [&] (const auto& lutItem) {
            itsParentLut.insert (make_pair (lutItem.first, lutItem.second->toHost ()));
            itsParentLutCounts.insert (make_pair (lutItem.first, lutItem.second->pointCount ()));
        });

        std::for_each (parentAveraging.cbegin (), parentAveraging.cend (), [&] (const auto& averagingItem) {
            itsAveraging.insert (make_pair (averagingItem.first, averagingItem.second->toHost ()));
        });
    }

    virtual void exportOctree (const std::string& path) = 0;

protected:
    uint32_t getRootIndex ()
    {
        return itsMetadata.nodeAmountSparse - 1;
    }

    bool isParentNode (uint32_t nodeIndex)
    {
        return this->itsOctree[nodeIndex].isParent;
    }

    bool isFinishedNode (uint32_t nodeIndex)
    {
        return this->itsOctree[nodeIndex].isFinished;
    }

    uint32_t getPointsInNode (uint32_t nodeIndex)
    {
        bool isParent = isParentNode (nodeIndex);
        return (isParent ? this->itsParentLutCounts[nodeIndex] : this->itsOctree[nodeIndex].pointCount);
    }

    uint32_t getChildNodeIndex (uint32_t nodeIndex, uint8_t child)
    {
        return this->itsOctree[nodeIndex].childrenChunks[child];
    }


protected:
    uint32_t itsPointsExported;
    uint32_t itsAbsorbedNodes;
    OctreeMetadata itsMetadata;
    uint8_t *itsCloud;
    unique_ptr<Chunk[]> itsOctree;
    unique_ptr<uint32_t[]> itsLeafeLut;
    unordered_map<uint32_t, unique_ptr<uint32_t[]>> itsParentLut;
    unordered_map<uint32_t, uint32_t> itsParentLutCounts;
    unordered_map<uint32_t, unique_ptr<Averaging[]>> itsAveraging;
};
