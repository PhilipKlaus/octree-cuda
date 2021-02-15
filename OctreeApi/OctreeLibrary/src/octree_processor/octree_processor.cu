//
// Created by KlausP on 04.11.2020.
//

#include "octree_processor.h"
#include "ply_exporter.cuh"
#include "potree_exporter.cuh"
#include "tools.cuh"

OctreeProcessor::OctreeProcessor (
        uint8_t* pointCloud,
        uint32_t chunkingGrid,
        uint32_t mergingThreshold,
        PointCloudMetadata cloudMetadata,
        SubsampleMetadata subsamplingMetadata)
{
    itsOctreeData = std::make_unique<OctreeData> (chunkingGrid);

    // Initialize metadata
    itsMetadata                  = {};
    itsMetadata.depth            = itsOctreeData->getDepth ();
    itsMetadata.nodeAmountDense  = itsOctreeData->getOverallNodes ();
    itsMetadata.chunkingGrid     = chunkingGrid;
    itsMetadata.mergingThreshold = mergingThreshold;
    itsMetadata.cloudMetadata    = cloudMetadata;
    itsSubsampleMetadata         = subsamplingMetadata;

    if (cloudMetadata.memoryType == CLOUD_HOST)
    {
        itsCloud = std::make_unique<PointCloudHost> (pointCloud, cloudMetadata);
    }
    else
    {
        itsCloud = std::make_unique<PointCloudDevice> (pointCloud, cloudMetadata);
    }

    // Create data LUT
    itsLeafLut = createGpuU32 (cloudMetadata.pointAmount, "Data LUT");
    spdlog::info ("Prepared empty SparseOctree");
}

void OctreeProcessor::calculateVoxelBB (PointCloudMetadata& metadata, uint32_t denseVoxelIndex, uint32_t level)
{
    Vector3<uint32_t> coords = {};

    // 1. Calculate coordinates of voxel within the actual level
    auto indexInLevel = denseVoxelIndex - itsOctreeData->getNodeOffset (level);
    tools::mapFromDenseIdxToDenseCoordinates (coords, indexInLevel, itsOctreeData->getGridSize (level));

    // 2. Calculate the bounding box for the actual voxel
    // ToDo: Include scale and offset!!!
    double min      = itsMetadata.cloudMetadata.bbCubic.min.x;
    double max      = itsMetadata.cloudMetadata.bbCubic.max.x;
    double side     = max - min;
    auto cubicWidth = side / itsOctreeData->getGridSize (level);

    metadata.bbCubic.min.x = itsMetadata.cloudMetadata.bbCubic.min.x + coords.x * cubicWidth;
    metadata.bbCubic.min.y = itsMetadata.cloudMetadata.bbCubic.min.y + coords.y * cubicWidth;
    metadata.bbCubic.min.z = itsMetadata.cloudMetadata.bbCubic.min.z + coords.z * cubicWidth;
    metadata.bbCubic.max.x = metadata.bbCubic.min.x + cubicWidth;
    metadata.bbCubic.max.y = metadata.bbCubic.min.y + cubicWidth;
    metadata.bbCubic.max.z = metadata.bbCubic.min.z + cubicWidth;
    metadata.cloudOffset   = metadata.bbCubic.min;
}

// ToDo: call appropriate export function!!!
void OctreeProcessor::exportPlyNodes (const string& folderPath)
{
    auto start = std::chrono::high_resolution_clock::now ();
    /*PlyExporter<coordinateType, colorType> plyExporter (
            itsCloudData, itsOctree, itsDataLUT, itsSubsampleLUTs, itsAveragingData, itsMetadata);
    plyExporter.exportOctree (folderPath);*/
    PotreeExporter<double, uint8_t> potreeExporter (
            itsCloud,
            itsOctreeData->getHost (),
            itsLeafLut,
            itsParentLut,
            itsAveragingData,
            itsMetadata,
            itsSubsampleMetadata);
    auto finish                           = std::chrono::high_resolution_clock::now ();
    std::chrono::duration<double> elapsed = finish - start;
    spdlog::info ("Copy from device to host tooks {} seconds", elapsed.count ());

    start = std::chrono::high_resolution_clock::now ();
    potreeExporter.exportOctree (folderPath);
    finish  = std::chrono::high_resolution_clock::now ();
    elapsed = finish - start;
    spdlog::info ("Export tooks {} seconds", elapsed.count ());
    itsTimeMeasurement.emplace_back ("exportPotree", elapsed.count () * 1000);
}
