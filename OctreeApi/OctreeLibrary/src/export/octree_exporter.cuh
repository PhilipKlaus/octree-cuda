#pragma once

#include "octree_metadata.h"
#include "types.cuh"


template <typename coordinateType, typename colorType>
class OctreeExporter
{
public:
    OctreeExporter (
            const GpuArrayU8& pointCloud,
            const GpuOctree& octree,
            const GpuArrayU32& leafeLut,
            const unordered_map<uint32_t, GpuArrayU32>& parentLut,
            const unordered_map<uint32_t, GpuAveraging>& parentAveraging,
            OctreeMetadata metadata) :
            itsMetadata (metadata),
            itsPointCloud (pointCloud->toHost ()), itsOctree (octree->toHost ()), itsLeafeLut (leafeLut->toHost ()),
            itsAbsorbedNodes (0)
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


protected:
    uint32_t itsAbsorbedNodes;
    OctreeMetadata itsMetadata;
    unique_ptr<uint8_t[]> itsPointCloud;
    unique_ptr<Chunk[]> itsOctree;
    unique_ptr<uint32_t[]> itsLeafeLut;
    unordered_map<uint32_t, unique_ptr<uint32_t[]>> itsParentLut;
    unordered_map<uint32_t, uint32_t> itsParentLutCounts;
    unordered_map<uint32_t, unique_ptr<Averaging[]>> itsAveraging;
};
