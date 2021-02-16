/**
 * @file octree_exporter-cuh
 * @author Philip Klaus
 * @brief Contains an octree-exporter base class definition
 */

#pragma once

#include "metadata.cuh"
#include "point_cloud.cuh"
#include "subsampling_data.cuh"
#include "types.cuh"


/**
 * OctreeExporter acts as a base class (abstract class) for several octree exporters.
 *
 * @tparam coordinateType The datatype of the 3D point coordinates.
 * @tparam colorType The datatype of the 3D point colors.
 */
template <typename coordinateType, typename colorType>
class OctreeExporter
{
public:
    OctreeExporter (
            const PointCloud& pointCloud,
            const shared_ptr<Chunk[]>& octree,
            const GpuArrayU32& leafLut,
            const std::shared_ptr<SubsamplingData>& subsamples,
            OctreeMetadata metadata,
            PointCloudMetadata cloudMetadata,
            SubsampleMetadata subsampleMetadata) :
            itsMetadata (metadata),
            itsCloudMetadata (cloudMetadata), itsSubsampleMetadata (subsampleMetadata),
            itsCloud (pointCloud->getCloudHost ()), itsOctree (octree), itsLeafLut (leafLut->toHost ()),
            itsAbsorbedNodes (0), itsPointsExported (0), itsSubsamples (subsamples)
    {}

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
        return (isParent ? this->itsSubsamples->getLutSize (nodeIndex) : this->itsOctree[nodeIndex].pointCount);
    }

    uint32_t getChildNodeIndex (uint32_t nodeIndex, uint8_t child)
    {
        return this->itsOctree[nodeIndex].childrenChunks[child];
    }

protected:
    uint32_t itsPointsExported;
    uint32_t itsAbsorbedNodes;
    OctreeMetadata itsMetadata;
    PointCloudMetadata itsCloudMetadata;
    SubsampleMetadata itsSubsampleMetadata;
    uint8_t* itsCloud;
    shared_ptr<Chunk[]> itsOctree;
    unique_ptr<uint32_t[]> itsLeafLut;
    std::shared_ptr<SubsamplingData> itsSubsamples;
};
