//
// Created by KlausP on 04.11.2020.
//

#include "hierarchical_merging.cuh"
#include "kernel_executor.cuh"
#include "octree_initialization.cuh"
#include "octree_processor.h"
#include "ply_exporter.cuh"
#include "point_count_propagation.cuh"
#include "point_counting.cuh"
#include "point_distributing.cuh"
#include "potree_exporter.cuh"


OctreeProcessor::OctreeProcessor (
        uint32_t chunkingGrid,
        uint32_t subsamplingGrid,
        uint32_t mergingThreshold,
        PointCloudMetadata cloudMetadata,
        SubsamplingStrategy strategy)
{
    // Initialize octree metadata
    itsMetadata                  = {};
    itsMetadata.depth            = tools::getOctreeLevel (chunkingGrid);
    itsMetadata.chunkingGrid     = chunkingGrid;
    itsMetadata.subsamplingGrid  = subsamplingGrid;
    itsMetadata.mergingThreshold = mergingThreshold;
    itsMetadata.cloudMetadata    = cloudMetadata;

    itsMetadata.strategy = strategy;

    // Pre calculate often-used octree metrics
    for (uint32_t gridSize = chunkingGrid; gridSize > 0; gridSize >>= 1)
    {
        itsGridSideLengthPerLevel.push_back (gridSize);
        itsLinearizedDenseVoxelOffset.push_back (itsMetadata.nodeAmountDense);
        itsVoxelsPerLevel.push_back (static_cast<uint32_t> (pow (gridSize, 3)));
        itsMetadata.nodeAmountDense += static_cast<uint32_t> (pow (gridSize, 3));
    }

    // Create data LUT
    itsLeafLut = createGpuU32 (cloudMetadata.pointAmount, "Data LUT");
    spdlog::info ("Prepared empty SparseOctree");
}

void OctreeProcessor::setPointCloudHost (uint8_t* pointCloud)
{
    itsCloudData = createGpuU8 (
            itsMetadata.cloudMetadata.pointAmount * itsMetadata.cloudMetadata.pointDataStride, "pointcloud");
    itsCloudData->toGPU (pointCloud);
    spdlog::info ("Copied point cloud from host->device");
}

void OctreeProcessor::setPointCloudDevice (uint8_t* pointCloud)
{
    itsCloudData = CudaArray<uint8_t>::fromDevicePtr (
            pointCloud,
            itsMetadata.cloudMetadata.pointAmount * itsMetadata.cloudMetadata.pointDataStride,
            "pointcloud");
    spdlog::info ("Imported point cloud from device");
}

void OctreeProcessor::setPointCloudDevice (GpuArrayU8 pointCloud)
{
    itsCloudData = std::move (pointCloud);
}

//###################
//#     Pipeline    #
//###################

void OctreeProcessor::initialPointCounting ()
{
    // Allocate the dense point count
    itsDensePointCountPerVoxel = createGpuU32 (itsMetadata.nodeAmountDense, "DensePointCountPerVoxel");
    itsDensePointCountPerVoxel->memset (0);

    // Allocate the conversion LUT from dense to sparse
    itsDenseToSparseLUT = createGpuI32 (itsMetadata.nodeAmountDense, "DenseToSparseLUT");
    itsDenseToSparseLUT->memset (-1);

    // Allocate the temporary sparseIndexCounter
    auto nodeAmountSparse = createGpuU32 (1, "nodeAmountSparse");
    nodeAmountSparse->memset (0);

    float time = Kernel::initialPointCounting (
            {
              itsMetadata.cloudMetadata.cloudType,
              itsMetadata.cloudMetadata.pointAmount
            },
            itsCloudData->devicePointer (),
            itsDensePointCountPerVoxel->devicePointer (),
            itsDenseToSparseLUT->devicePointer (),
            nodeAmountSparse->devicePointer (),
            itsMetadata.cloudMetadata,
            itsGridSideLengthPerLevel[0]);

    // Store the current amount of sparse nodes
    // !IMPORTANT! At this time nodeAmountSparse holds just the amount of nodes
    // in the lowest level
    itsMetadata.nodeAmountSparse = nodeAmountSparse->toHost ()[0];
    itsTimeMeasurement.emplace_back ("initialPointCount", time);
    spdlog::info ("'initialPointCounting' took {:f} [ms]", time);
}


void OctreeProcessor::performCellMerging ()
{
    // Allocate the temporary sparseIndexCounter
    auto nodeAmountSparse = createGpuU32 (1, "nodeAmountSparse");
    // !IMPORTANT! initialize it with the current sparse node counts (from base level)
    gpuErrchk (cudaMemcpy (
            nodeAmountSparse->devicePointer (),
            &itsMetadata.nodeAmountSparse,
            1 * sizeof (uint32_t),
            cudaMemcpyHostToDevice));

    float timeAccumulated = 0;

    // Perform a hierarchicaly merging of the grid cells which results in an octree structure
    for (uint32_t i = 0; i < itsMetadata.depth; ++i)
    {
        float time = executeKernel (
                chunking::kernelPropagatePointCounts,
                itsVoxelsPerLevel[i + 1],
                itsDensePointCountPerVoxel->devicePointer (),
                itsDenseToSparseLUT->devicePointer (),
                nodeAmountSparse->devicePointer (),
                itsVoxelsPerLevel[i + 1],
                itsGridSideLengthPerLevel[i + 1],
                itsGridSideLengthPerLevel[i],
                itsLinearizedDenseVoxelOffset[i + 1],
                itsLinearizedDenseVoxelOffset[i]);

        itsTimeMeasurement.emplace_back ("propagatePointCounts_" + std::to_string (itsGridSideLengthPerLevel[i]), time);
        timeAccumulated += time;
    }

    spdlog::info ("'propagatePointCounts' took {:f} [ms]", timeAccumulated);

    // Retrieve the actual amount of sparse nodes in the octree and allocate the octree data structure
    itsMetadata.nodeAmountSparse = nodeAmountSparse->toHost ()[0];
    itsOctree                    = createGpuOctree (itsMetadata.nodeAmountSparse, "octreeSparse");

    spdlog::info (
            "Sparse octree ({} voxels) -> Memory saving: {:f} [%] {:f} [GB]",
            itsMetadata.nodeAmountSparse,
            (1 - static_cast<float> (itsMetadata.nodeAmountSparse) / itsMetadata.nodeAmountDense) * 100,
            static_cast<float> (itsMetadata.nodeAmountDense - itsMetadata.nodeAmountSparse) * sizeof (Chunk) /
                    1000000000.f);

    // Allocate the conversion LUT from sparse to dense
    itsSparseToDenseLUT = createGpuI32 (itsMetadata.nodeAmountSparse, "sparseToDenseLUT");
    itsSparseToDenseLUT->memset (-1);

    initLowestOctreeHierarchy ();
    mergeHierarchical ();
}


void OctreeProcessor::initLowestOctreeHierarchy ()
{
    float time = executeKernel (
            chunking::kernelOctreeInitialization,
            itsVoxelsPerLevel[0],
            itsOctree->devicePointer (),
            itsDensePointCountPerVoxel->devicePointer (),
            itsDenseToSparseLUT->devicePointer (),
            itsSparseToDenseLUT->devicePointer (),
            itsVoxelsPerLevel[0]);

    itsTimeMeasurement.emplace_back ("initLowestOctreeHierarchy", time);
    spdlog::info ("'initLowestOctreeHierarchy' took {:f} [ms]", time);
}


void OctreeProcessor::mergeHierarchical ()
{
    // Create a temporary counter register for assigning indices for chunks within the 'itsDataLUT' register
    auto globalChunkCounter = createGpuU32 (1, "globalChunkCounter");
    globalChunkCounter->memset (0);

    // Perform a hierarchicaly merging of the grid cells which results in an octree structure
    float timeAccumulated = 0;
    for (uint32_t i = 0; i < itsMetadata.depth; ++i)
    {
        float time = executeKernel (
                chunking::kernelMergeHierarchical,
                itsVoxelsPerLevel[i + 1],
                itsOctree->devicePointer (),
                itsDensePointCountPerVoxel->devicePointer (),
                itsDenseToSparseLUT->devicePointer (),
                itsSparseToDenseLUT->devicePointer (),
                globalChunkCounter->devicePointer (),
                itsMetadata.mergingThreshold,
                itsVoxelsPerLevel[i + 1],
                itsGridSideLengthPerLevel[i + 1],
                itsGridSideLengthPerLevel[i],
                itsLinearizedDenseVoxelOffset[i + 1],
                itsLinearizedDenseVoxelOffset[i]);

        timeAccumulated += time;
        itsTimeMeasurement.emplace_back ("mergeHierarchical_" + std::to_string (itsGridSideLengthPerLevel[i]), time);
    }

    spdlog::info ("'mergeHierarchical' took {:f} [ms]", timeAccumulated);
}


void OctreeProcessor::distributePoints ()
{
    // Create temporary indexRegister for assigning an index for each point within its chunk area
    auto tmpIndexRegister = createGpuU32 (itsMetadata.nodeAmountSparse, "tmpIndexRegister");
    tmpIndexRegister->memset (0);

    float time = Kernel::distributePoints (
            {
              itsMetadata.cloudMetadata.cloudType,
              itsMetadata.cloudMetadata.pointAmount
            },
            itsOctree->devicePointer (),
            itsCloudData->devicePointer (),
            itsLeafLut->devicePointer (),
            itsDenseToSparseLUT->devicePointer (),
            tmpIndexRegister->devicePointer (),
            itsMetadata.cloudMetadata,
            itsGridSideLengthPerLevel[0]);

    itsTimeMeasurement.emplace_back ("distributePointsSparse", time);
    spdlog::info ("'distributePoints' took {:f} [ms]", time);
}


void OctreeProcessor::performSubsampling ()
{
    auto h_octreeSparse     = itsOctree->toHost ();
    auto h_sparseToDenseLUT = itsSparseToDenseLUT->toHost ();
    auto nodesBaseLevel     = static_cast<uint32_t> (pow (itsMetadata.subsamplingGrid, 3.f));

    // Prepare data strucutres for the subsampling
    auto pointCountGrid  = createGpuU32 (nodesBaseLevel, "pointCountGrid");
    auto denseToSpareLUT = createGpuI32 (nodesBaseLevel, "denseToSpareLUT");
    auto voxelCount      = createGpuU32 (1, "voxelCount");
    auto subsampleData   = createGpuSubsample (8, "subsampleData");

    pointCountGrid->memset (0);
    denseToSpareLUT->memset (-1);
    voxelCount->memset (0);

    SubsamplingTimings timings = {};

    if (itsMetadata.strategy == RANDOM_POINT)
    {
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
    }
    else
    {
        timings = firstPointSubsampling (
                h_octreeSparse,
                h_sparseToDenseLUT,
                getRootIndex (),
                itsMetadata.depth,
                pointCountGrid,
                denseToSpareLUT,
                voxelCount,
                subsampleData);
    }


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
        SubsampleSet &subsampleSet,
        Chunk& voxel,
        const unique_ptr<Chunk[]>& h_octreeSparse,
        uint32_t& accumulatedPoints)
{
    uint32_t i = 0;
    for (int childIndex : voxel.childrenChunks)
    {
        if (childIndex != -1)
        {
            SubsampleConfig *config;
            switch(i) {
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
            config->averagingAdress =
                    child.isParent ? itsAveragingData[childIndex]->devicePointer () : nullptr;
            config->lutStartIndex    = child.isParent ? 0 : child.chunkDataIndex;
            config->pointOffsetLower = accumulatedPoints;
            accumulatedPoints += child.isParent ? itsParentLut[childIndex]->pointCount () : child.pointCount;
            config->pointOffsetUpper = accumulatedPoints;
        }
        ++i;
    }
}

void OctreeProcessor::calculateVoxelBB (
        PointCloudMetadata& metadata, uint32_t denseVoxelIndex, uint32_t level)
{
    Vector3<uint32_t> coords = {};

    // 1. Calculate coordinates of voxel within the actual level
    auto indexInLevel = denseVoxelIndex - itsLinearizedDenseVoxelOffset[level];
    tools::mapFromDenseIdxToDenseCoordinates (coords, indexInLevel, itsGridSideLengthPerLevel[level]);

    // 2. Calculate the bounding box for the actual voxel
    // ToDo: Include scale and offset!!!
    double min  = itsMetadata.cloudMetadata.bbCubic.min.x;
    double max  = itsMetadata.cloudMetadata.bbCubic.max.x;
    double side = max - min;
    auto cubicWidth     = side / itsGridSideLengthPerLevel[level];

    metadata.bbCubic.min.x = itsMetadata.cloudMetadata.bbCubic.min.x + coords.x * cubicWidth;
    metadata.bbCubic.min.y = itsMetadata.cloudMetadata.bbCubic.min.y + coords.y * cubicWidth;
    metadata.bbCubic.min.z = itsMetadata.cloudMetadata.bbCubic.min.z + coords.z * cubicWidth;
    metadata.bbCubic.max.x = metadata.bbCubic.min.x + cubicWidth;
    metadata.bbCubic.max.y = metadata.bbCubic.min.y + cubicWidth;
    metadata.bbCubic.max.z = metadata.bbCubic.min.z + cubicWidth;
    metadata.cloudOffset   = metadata.bbCubic.min;
}

void OctreeProcessor::exportPlyNodes (const string& folderPath)
{
    /*PlyExporter<coordinateType, colorType> plyExporter (
            itsCloudData, itsOctree, itsDataLUT, itsSubsampleLUTs, itsAveragingData, itsMetadata);
    plyExporter.exportOctree (folderPath);*/
    PotreeExporter<double, uint8_t > potreeExporter (
            itsCloudData, itsOctree, itsLeafLut, itsParentLut, itsAveragingData, itsMetadata);
    potreeExporter.exportOctree (folderPath);
}
