//
// Created by KlausP on 04.11.2020.
//

#include "octree_processor_impl.cuh"
#include "ply_exporter.cuh"
#include "potree_exporter.cuh"
#include "time_tracker.cuh"
#include "tools.cuh"

OctreeProcessor::OctreeProcessorImpl::OctreeProcessorImpl (
        uint8_t* pointCloud,
        uint32_t chunkingGrid,
        uint32_t mergingThreshold,
        PointCloudMetadata cloudMetadata,
        SubsampleMetadata subsamplingMetadata)
{
    itsOctreeData = std::make_unique<Octree> (chunkingGrid);

    // ToDo: Move itsMeatadata to OctreeData
    // Initialize metadata
    itsMetadata                  = {};
    itsMetadata.depth            = itsOctreeData->getDepth ();
    itsMetadata.nodeAmountDense  = itsOctreeData->getOverallNodes ();
    itsMetadata.chunkingGrid     = chunkingGrid;
    itsMetadata.mergingThreshold = mergingThreshold;
    itsSubsampleMetadata         = subsamplingMetadata;

    if (cloudMetadata.memoryType == CLOUD_HOST)
    {
        itsCloud = std::make_unique<PointCloudHost> (pointCloud, cloudMetadata);
    }
    else
    {
        itsCloud = std::make_unique<PointCloudDevice> (pointCloud, cloudMetadata);
    }

    //-----------------------------
    // Create GPU data for chunking
    //-----------------------------

    auto timing = Timing::TimeTracker::start ();

    // Allocate the dense point count
    itsCountingGrid = createGpuU32 (itsMetadata.nodeAmountDense, "countingGrid");
    itsCountingGrid->memset (0);

    // Allocate the conversion LUT from dense to sparse
    itsDenseToSparseLUT = createGpuI32 (itsMetadata.nodeAmountDense, "denseToSparseLut");
    itsDenseToSparseLUT->memset (-1);

    // Allocate the temporary sparseIndexCounter
    itsTmpCounting = createGpuU32 (1, "tmpCounting");
    itsTmpCounting->memset (0);

    itsSubsamples       = std::make_shared<SubsamplingData> (itsSubsampleMetadata.subsamplingGrid);

    auto expectedPoints = static_cast<uint32_t> (itsCloud->getMetadata ().pointAmount * 2.1);
    itsPointLut = createGpuOutputData (expectedPoints, "pointLUT");
    itsPointLut->memset (0);

    Timing::TimeTracker::stop (timing, "Preparing GPU data", Timing::Time::PROCESS);
}

void OctreeProcessor::OctreeProcessorImpl::calculateVoxelBB (
        PointCloudMetadata& metadata, uint32_t denseVoxelIndex, uint32_t level)
{
    Vector3<uint32_t> coords = {};

    // 1. Calculate coordinates of voxel within the actual level
    auto indexInLevel = denseVoxelIndex - itsOctreeData->getNodeOffset (level);
    tools::mapFromDenseIdxToDenseCoordinates (coords, indexInLevel, itsOctreeData->getGridSize (level));

    // 2. Calculate the bounding box for the actual voxel
    // ToDo: Include scale and offset!!!
    auto& cloudMeta = itsCloud->getMetadata ();
    double min      = cloudMeta.bbCubic.min.x;
    double max      = cloudMeta.bbCubic.max.x;
    double side     = max - min;
    auto cubicWidth = side / itsOctreeData->getGridSize (level);

    metadata.bbCubic.min.x = cloudMeta.bbCubic.min.x + coords.x * cubicWidth;
    metadata.bbCubic.min.y = cloudMeta.bbCubic.min.y + coords.y * cubicWidth;
    metadata.bbCubic.min.z = cloudMeta.bbCubic.min.z + coords.z * cubicWidth;
    metadata.bbCubic.max.x = metadata.bbCubic.min.x + cubicWidth;
    metadata.bbCubic.max.y = metadata.bbCubic.min.y + cubicWidth;
    metadata.bbCubic.max.z = metadata.bbCubic.min.z + cubicWidth;
    metadata.cloudOffset   = metadata.bbCubic.min;
}

void OctreeProcessor::OctreeProcessorImpl::exportPotree (const string& folderPath)
{
    itsOctreeData->copyToHost();

    if (itsCloud->getMetadata ().cloudType == CLOUD_FLOAT_UINT8_T)
    {
        PotreeExporter<float, uint8_t> potreeExporter (
                itsCloud,
                itsOctreeData->getHost (),
                itsSubsamples,
                itsMetadata,
                itsCloud->getMetadata (),
                itsSubsampleMetadata);
        potreeExporter.exportOctree (folderPath);
    }
    else
    {
        PotreeExporter<double, uint8_t> potreeExporter (
                itsCloud,
                itsOctreeData->getHost (),
                itsSubsamples,
                itsMetadata,
                itsCloud->getMetadata (),
                itsSubsampleMetadata);
        potreeExporter.exportOctree (folderPath);
    }
}

void OctreeProcessor::OctreeProcessorImpl::exportPlyNodes (const string& folderPath)
{
    itsOctreeData->copyToHost();

    if (itsCloud->getMetadata ().cloudType == CLOUD_FLOAT_UINT8_T)
    {
        PlyExporter<float, uint8_t> plyExporter (
                itsCloud,
                itsOctreeData->getHost (),
                itsSubsamples,
                itsMetadata,
                itsCloud->getMetadata (),
                itsSubsampleMetadata);
        plyExporter.exportOctree (folderPath);
    }
    else
    {
        PotreeExporter<double, uint8_t> plyExporter (
                itsCloud,
                itsOctreeData->getHost (),
                itsSubsamples,
                itsMetadata,
                itsCloud->getMetadata (),
                itsSubsampleMetadata);
        plyExporter.exportOctree (folderPath);
    }
}
