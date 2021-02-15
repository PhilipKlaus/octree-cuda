//
// Created by KlausP on 04.11.2020.
//

#include "octree_processor.h"
#include "octree_processpr_impl.cuh"
#include "ply_exporter.cuh"
#include "potree_exporter.cuh"
#include "tools.cuh"

OctreeProcessorPimpl::OctreeProcessorPimpl (
        uint8_t* pointCloud,
        uint32_t chunkingGrid,
        uint32_t mergingThreshold,
        PointCloudMetadata cloudMetadata,
        SubsampleMetadata subsamplingMetadata) {
 itsProcessor = std::make_unique<OctreeProcessorPimpl::OctreeProcessorImpl>(pointCloud, chunkingGrid, mergingThreshold, cloudMetadata, subsamplingMetadata);
}

void OctreeProcessorPimpl::initialPointCounting () {
    itsProcessor->initialPointCounting();
}

void OctreeProcessorPimpl::performCellMerging () {
    itsProcessor->performCellMerging();
}
void OctreeProcessorPimpl::distributePoints () {
    itsProcessor->distributePoints();
}
void OctreeProcessorPimpl::performSubsampling () {
    itsProcessor->performSubsampling();
}

void OctreeProcessorPimpl::exportPlyNodes (const std::string& folderPath)
{
    itsProcessor->exportPlyNodes(folderPath);
}

void OctreeProcessorPimpl::exportPotree (const std::string& folderPath)
{
    itsProcessor->exportPlyNodes(folderPath);
}
void OctreeProcessorPimpl::updateStatistics ()
{
    itsProcessor->updateOctreeStatistics();
}
const std::vector<std::tuple<std::string, float>>& OctreeProcessorPimpl::getTimings ()
{
    return itsProcessor->getTimings();
}
void OctreeProcessorPimpl::exportHistogram (const std::string& filePath, uint32_t binWidth)
{
    itsProcessor->exportHistogram(filePath, binWidth);
}

const OctreeMetadata& OctreeProcessorPimpl::getOctreeMetadata ()
{
    return itsProcessor->getMetadata();
}

OctreeProcessorPimpl::~OctreeProcessorPimpl ()
{
    // Empty destructor - necessary becaus of PIMPL
    // https://stackoverflow.com/questions/9954518/stdunique-ptr-with-an-incomplete-type-wont-compile
}

OctreeProcessorPimpl::OctreeProcessorImpl::OctreeProcessorImpl (
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

void OctreeProcessorPimpl::OctreeProcessorImpl::calculateVoxelBB (PointCloudMetadata& metadata, uint32_t denseVoxelIndex, uint32_t level)
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

// ToDo: call appropriate export function!!!
void OctreeProcessorPimpl::OctreeProcessorImpl::exportPlyNodes (const string& folderPath)
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
            itsCloud->getMetadata (),
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
