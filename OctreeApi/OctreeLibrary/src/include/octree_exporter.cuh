/**
 * @file octree_exporter-cuh
 * @author Philip Klaus
 * @brief Contains an octree-exporter base class definition
 */

#pragma once

#include "metadata.cuh"
#include "point_cloud.cuh"
#include "types.cuh"


/**
 * OctreeExporter acts as a base class (abstract class) for several octree exporters.
 */
class OctreeExporter
{
public:
    OctreeExporter (
            const PointCloud& pointCloud,
            const shared_ptr<Chunk[]>& octree,
            OctreeMetadata metadata,
            PointCloudMetadata cloudMetadata,
            SubsampleMetadata subsampleMetadata) :
            itsMetadata (metadata),
            itsCloudMetadata (cloudMetadata), itsSubsampleMetadata (subsampleMetadata), itsOctree (octree),
            itsOutputBufferSize (pointCloud->getOutputBufferSize ()), itsPointsExported (0),
            itsOutputBuffer (std::move (pointCloud->getOutputBuffer_h ()))
    {}

    virtual void exportOctree (const std::string& path) = 0;

protected:
    uint32_t getRootIndex ()
    {
        return itsMetadata.nodeAmountSparse - 1;
    }

    bool isFinishedNode (uint32_t nodeIndex)
    {
        return this->itsOctree[nodeIndex].isFinished;
    }

    uint32_t getPointsInNode (uint32_t nodeIndex)
    {
        return this->itsOctree[nodeIndex].pointCount;
    }

    uint32_t getChildNodeIndex (uint32_t nodeIndex, uint8_t child)
    {
        return this->itsOctree[nodeIndex].childrenChunks[child];
    }

    uint64_t getDataIndex (uint32_t nodeIndex)
    {
        return this->itsOctree[nodeIndex].chunkDataIndex;
    }

protected:
    uint32_t itsPointsExported;
    OctreeMetadata itsMetadata;
    PointCloudMetadata itsCloudMetadata;
    SubsampleMetadata itsSubsampleMetadata;
    shared_ptr<Chunk[]> itsOctree;
    std::unique_ptr<OutputBuffer[]> itsOutputBuffer;
    uint64_t itsOutputBufferSize;
};
