//
// Created by KlausP on 04.11.2020.
//

#include <sparseOctree.h>

#include <hierarchical_merging.cuh>
#include <kernel_executor.cuh>
#include <octree_initialization.cuh>
#include <point_count_propagation.cuh>
#include <point_counting.cuh>
#include <point_distributing.cuh>


template <typename coordinateType, typename colorType>
SparseOctree<coordinateType, colorType>::SparseOctree (
        GridSize chunkingGrid,
        GridSize subsamplingGrid,
        uint32_t mergingThreshold,
        PointCloudMetadata cloudMetadata,
        unique_ptr<CudaArray<uint8_t>> cloudData,
        SubsamplingStrategy strategy) :

        itsCloudData (move (cloudData))
{
    // Initialize octree metadata
    itsMetadata                  = {};
    itsMetadata.depth            = tools::getOctreeLevel (chunkingGrid);
    itsMetadata.chunkingGrid     = chunkingGrid;
    itsMetadata.subsamplingGrid  = subsamplingGrid;
    itsMetadata.mergingThreshold = mergingThreshold;
    itsMetadata.cloudMetadata    = cloudMetadata;
    itsMetadata.strategy         = strategy;

    // Pre calculate often-used octree metrics
    auto sideLength = static_cast<uint32_t> (pow (2, itsMetadata.depth));
    for (uint32_t gridSize = sideLength; gridSize > 0; gridSize >>= 1)
    {
        itsGridSideLengthPerLevel.push_back (gridSize);
        itsLinearizedDenseVoxelOffset.push_back (itsMetadata.nodeAmountDense);
        itsVoxelsPerLevel.push_back (static_cast<uint32_t> (pow (gridSize, 3)));
        itsMetadata.nodeAmountDense += static_cast<uint32_t> (pow (gridSize, 3));
    }

    // Create data LUT
    itsDataLUT = createGpuU32 (cloudMetadata.pointAmount, "Data LUT");
    spdlog::info ("Instantiated sparse octree for {} points", cloudMetadata.pointAmount);
}


//###################
//#     Pipeline    #
//###################

template <typename coordinateType, typename colorType>
void SparseOctree<coordinateType, colorType>::initialPointCounting ()
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

    float time = executeKernel (
            chunking::kernelInitialPointCounting<float>,
            itsMetadata.cloudMetadata.pointAmount,
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


template <typename coordinateType, typename colorType>
void SparseOctree<coordinateType, colorType>::performCellMerging ()
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


template <typename coordinateType, typename colorType>
void SparseOctree<coordinateType, colorType>::initLowestOctreeHierarchy ()
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


template <typename coordinateType, typename colorType>
void SparseOctree<coordinateType, colorType>::mergeHierarchical ()
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


template <typename coordinateType, typename colorType>
void SparseOctree<coordinateType, colorType>::distributePoints ()
{
    // Create temporary indexRegister for assigning an index for each point within its chunk area
    auto tmpIndexRegister = createGpuU32 (itsMetadata.nodeAmountSparse, "tmpIndexRegister");
    tmpIndexRegister->memset (0);

    float time = executeKernel (
            chunking::kernelDistributePoints<float>,
            itsMetadata.cloudMetadata.pointAmount,
            itsOctree->devicePointer (),
            itsCloudData->devicePointer (),
            itsDataLUT->devicePointer (),
            itsDenseToSparseLUT->devicePointer (),
            tmpIndexRegister->devicePointer (),
            itsMetadata.cloudMetadata,
            itsGridSideLengthPerLevel[0]);

    itsTimeMeasurement.emplace_back ("distributePointsSparse", time);
    spdlog::info ("'distributePoints' took {:f} [ms]", time);
}


template <typename coordinateType, typename colorType>
void SparseOctree<coordinateType, colorType>::performSubsampling ()
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

    std::tuple<float, float> time (0.f, 0.f);

    if (itsMetadata.strategy == RANDOM_POINT)
    {
        auto randomStates = createGpuRandom (1024, "randomStates");

        // ToDo: Time measurement
        initRandomStates (std::time (0), randomStates, 1024);
        auto randomIndices = createGpuU32 (nodesBaseLevel, "randomIndices");

        time = randomSubsampling (
                h_octreeSparse,
                h_sparseToDenseLUT,
                getRootIndex (),
                itsMetadata.depth,
                pointCountGrid,
                denseToSpareLUT,
                voxelCount,
                randomStates,
                randomIndices,
                subsampleData);
    }
    else
    {
        time = firstPointSubsampling (
                h_octreeSparse,
                h_sparseToDenseLUT,
                getRootIndex (),
                itsMetadata.depth,
                pointCountGrid,
                denseToSpareLUT,
                voxelCount,
                subsampleData);
    }


    itsTimeMeasurement.emplace_back ("subsampleEvaluation", get<0> (time));
    itsTimeMeasurement.emplace_back ("subsampling", get<1> (time));
    spdlog::info ("subsample evaluation took {}[ms]", get<0> (time));
    spdlog::info ("subsampling took {}[ms]", get<1> (time));
}


template <typename coordinateType, typename colorType>
void SparseOctree<coordinateType, colorType>::prepareSubsampleConfig (
        Chunk& voxel,
        const unique_ptr<Chunk[]>& h_octreeSparse,
        GpuSubsample& subsampleData,
        uint32_t& accumulatedPoints)
{
    // Prepare subsample data and copy it to the GPU
    SubsampleConfig newSubsampleData[8];
    uint32_t i = 0;
    for (int childIndex : voxel.childrenChunks)
    {
        if (childIndex != -1)
        {
            Chunk child = h_octreeSparse[childIndex];
            newSubsampleData[i].lutAdress =
                    child.isParent ? itsSubsampleLUTs[childIndex]->devicePointer () : itsDataLUT->devicePointer ();
            newSubsampleData[i].averagingAdress =
                    child.isParent ? itsAveragingData[childIndex]->devicePointer () : nullptr;
            newSubsampleData[i].lutStartIndex    = child.isParent ? 0 : child.chunkDataIndex;
            newSubsampleData[i].pointOffsetLower = accumulatedPoints;
            accumulatedPoints += child.isParent ? itsSubsampleLUTs[childIndex]->pointCount () : child.pointCount;
            newSubsampleData[i].pointOffsetUpper = accumulatedPoints;
            ++i;
        }
    }
    subsampleData->toGPU (reinterpret_cast<uint8_t*> (newSubsampleData));
}

template <typename coordinateType, typename colorType>
void SparseOctree<coordinateType, colorType>::calculateVoxelBB (
        PointCloudMetadata& metadata, uint32_t denseVoxelIndex, uint32_t level)
{
    Vector3<uint32_t> coords = {};

    // 1. Calculate coordinates of voxel within the actual level
    auto indexInLevel = denseVoxelIndex - itsLinearizedDenseVoxelOffset[level];
    tools::mapFromDenseIdxToDenseCoordinates (coords, indexInLevel, itsGridSideLengthPerLevel[level]);

    // 2. Calculate the bounding box for the actual voxel
    // ToDo: Include scale and offset!!!
    float side      = itsMetadata.cloudMetadata.boundingBox.maximum.x - itsMetadata.cloudMetadata.boundingBox.minimum.x;
    auto cubicWidth = side / static_cast<float> (itsGridSideLengthPerLevel[level]);

    metadata.boundingBox.minimum.x = itsMetadata.cloudMetadata.boundingBox.minimum.x + coords.x * cubicWidth;
    metadata.boundingBox.minimum.y = itsMetadata.cloudMetadata.boundingBox.minimum.y + coords.y * cubicWidth;
    metadata.boundingBox.minimum.z = itsMetadata.cloudMetadata.boundingBox.minimum.z + coords.z * cubicWidth;
    metadata.boundingBox.maximum.x = metadata.boundingBox.minimum.x + cubicWidth;
    metadata.boundingBox.maximum.y = metadata.boundingBox.minimum.y + cubicWidth;
    metadata.boundingBox.maximum.z = metadata.boundingBox.minimum.z + cubicWidth;
    metadata.cloudOffset           = metadata.boundingBox.minimum;
}


//----------------------------------------------------------------------------------------------------------------------
//                                           SparseOctree<float, uint8_t>
//----------------------------------------------------------------------------------------------------------------------
template SparseOctree<float, uint8_t>::SparseOctree (
        GridSize chunkingGrid,
        GridSize subsamplingGrid,
        uint32_t mergingThreshold,
        PointCloudMetadata cloudMetadata,
        unique_ptr<CudaArray<uint8_t>> cloudData,
        SubsamplingStrategy strategy);
template void SparseOctree<float, uint8_t>::initialPointCounting ();
template void SparseOctree<float, uint8_t>::performCellMerging ();
template void SparseOctree<float, uint8_t>::distributePoints ();
template void SparseOctree<float, uint8_t>::performSubsampling ();
template void SparseOctree<float, uint8_t>::initLowestOctreeHierarchy ();
template void SparseOctree<float, uint8_t>::mergeHierarchical ();
template void SparseOctree<float, uint8_t>::prepareSubsampleConfig (
        Chunk& voxel,
        const unique_ptr<Chunk[]>& h_octreeSparse,
        GpuSubsample& subsampleData,
        uint32_t& accumulatedPoints);
template void SparseOctree<float, uint8_t>::calculateVoxelBB (
        PointCloudMetadata& metadata, uint32_t denseVoxelIndex, uint32_t level);

//----------------------------------------------------------------------------------------------------------------------
//                                           SparseOctree<double, uint16_t>
//----------------------------------------------------------------------------------------------------------------------
template SparseOctree<double, uint16_t>::SparseOctree (
        GridSize chunkingGrid,
        GridSize subsamplingGrid,
        uint32_t mergingThreshold,
        PointCloudMetadata cloudMetadata,
        unique_ptr<CudaArray<uint8_t>> cloudData,
        SubsamplingStrategy strategy);
template void SparseOctree<double, uint16_t>::initialPointCounting ();
template void SparseOctree<double, uint16_t>::performCellMerging ();
template void SparseOctree<double, uint16_t>::distributePoints ();
template void SparseOctree<double, uint16_t>::performSubsampling ();
template void SparseOctree<double, uint16_t>::initLowestOctreeHierarchy ();
template void SparseOctree<double, uint16_t>::mergeHierarchical ();
template void SparseOctree<double, uint16_t>::prepareSubsampleConfig (
        Chunk& voxel,
        const unique_ptr<Chunk[]>& h_octreeSparse,
        GpuSubsample& subsampleData,
        uint32_t& accumulatedPoints);
template void SparseOctree<double, uint16_t>::calculateVoxelBB (
        PointCloudMetadata& metadata, uint32_t denseVoxelIndex, uint32_t level);
