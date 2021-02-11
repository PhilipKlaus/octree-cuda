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


void OctreeProcessor::performSubsampling ()
{
    auto h_octreeSparse     = itsOctree->toHost ();
    auto h_sparseToDenseLUT = itsSparseToDenseLUT->toHost ();
    auto nodesBaseLevel     = static_cast<uint32_t> (pow (itsSubsampleMetadata.subsamplingGrid, 3.f));

    // Prepare data strucutres for the subsampling
    auto pointCountGrid  = createGpuU32 (nodesBaseLevel, "pointCountGrid");
    auto denseToSpareLUT = createGpuI32 (nodesBaseLevel, "denseToSpareLUT");
    auto voxelCount      = createGpuU32 (1, "voxelCount");
    auto subsampleData   = createGpuSubsample (8, "subsampleData");

    pointCountGrid->memset (0);
    denseToSpareLUT->memset (-1);
    voxelCount->memset (0);

    SubsamplingTimings timings = {};

    auto randomStates = createGpuRandom (1024, "randomStates");

    // ToDo: Time measurement
    initRandomStates (std::time (0), randomStates, 1024);
    auto randomIndices = createGpuU32 (nodesBaseLevel, "randomIndices");

    timings = randomSubsampling (
            h_octreeSparse,
            h_sparseToDenseLUT,
            getRootIndex (),
            itsMetadata.depth,
            pointCountGrid,
            denseToSpareLUT,
            voxelCount,
            randomStates,
            randomIndices);


    itsTimeMeasurement.emplace_back ("subsampleEvaluation", timings.subsampleEvaluation);
    itsTimeMeasurement.emplace_back ("generateRandoms", timings.generateRandoms);
    itsTimeMeasurement.emplace_back ("averaging", timings.averaging);
    itsTimeMeasurement.emplace_back ("subsampling", timings.subsampling);
    spdlog::info ("subsample evaluation took {}[ms]", timings.subsampleEvaluation);
    spdlog::info ("generateRandoms took {}[ms]", timings.generateRandoms);
    spdlog::info ("averaging took {}[ms]", timings.averaging);
    spdlog::info ("subsampling took {}[ms]", timings.subsampling);
}


void OctreeProcessor::prepareSubsampleConfig (
        SubsampleSet& subsampleSet,
        Chunk& voxel,
        const unique_ptr<Chunk[]>& h_octreeSparse,
        uint32_t& accumulatedPoints)
{
    uint32_t i = 0;
    for (int childIndex : voxel.childrenChunks)
    {
        if (childIndex != -1)
        {
            SubsampleConfig* config;
            switch (i)
            {
            case 0:
                config = &subsampleSet.child_0;
                break;
            case 1:
                config = &subsampleSet.child_1;
                break;
            case 2:
                config = &subsampleSet.child_2;
                break;
            case 3:
                config = &subsampleSet.child_3;
                break;
            case 4:
                config = &subsampleSet.child_4;
                break;
            case 5:
                config = &subsampleSet.child_5;
                break;
            case 6:
                config = &subsampleSet.child_6;
                break;
            default:
                config = &subsampleSet.child_7;
                break;
            }
            Chunk child = h_octreeSparse[childIndex];
            config->lutAdress =
                    child.isParent ? itsParentLut[childIndex]->devicePointer () : itsLeafLut->devicePointer ();
            config->averagingAdress  = child.isParent ? itsAveragingData[childIndex]->devicePointer () : nullptr;
            config->lutStartIndex    = child.isParent ? 0 : child.chunkDataIndex;
            config->pointOffsetLower = accumulatedPoints;
            accumulatedPoints += child.isParent ? itsParentLut[childIndex]->pointCount () : child.pointCount;
            config->pointOffsetUpper = accumulatedPoints;
        }
        ++i;
    }
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
            itsCloud, itsOctree, itsLeafLut, itsParentLut, itsAveragingData, itsMetadata, itsSubsampleMetadata);
    potreeExporter.exportOctree (folderPath);


    auto finish                           = std::chrono::high_resolution_clock::now ();
    std::chrono::duration<double> elapsed = finish - start;
    spdlog::info ("Export tooks {} seconds", elapsed.count ());
    itsTimeMeasurement.emplace_back ("exportPotree", elapsed.count () * 1000);
}
