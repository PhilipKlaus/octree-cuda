/**
 * @file octree_exporter-cuh
 * @author Philip Klaus
 * @brief Contains an octree-exporter base class definition
 */

#pragma once

#include "metadata.cuh"
#include "octree.cuh"
#include "point_cloud.cuh"
#include "types.cuh"


/**
 * OctreeExporter acts as a base class (abstract class) for several octree exporters.
 */
class OctreeExporter
{
public:
    OctreeExporter () : itsPointsExported (0)
    {}

    virtual void exportOctree (
            const std::string& path,
            const PointCloud& pointCloud,
            const Octree& octree,
            const ProcessingInfo& subsampleMetadata) = 0;

protected:
    uint32_t itsPointsExported;
};
