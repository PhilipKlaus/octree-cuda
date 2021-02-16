#include "kernel_executor.cuh"
#include "octree_processor_impl.cuh"
#include "random_subsampling.cuh"
#include "subsample_evaluating.cuh"


void OctreeProcessor::OctreeProcessorImpl::performSubsampling ()
{
    auto h_octreeSparse     = itsOctreeData->getHost ();
    auto h_sparseToDenseLUT = itsSparseToDenseLUT->toHost ();
    auto nodesBaseLevel     = static_cast<uint32_t> (pow (itsSubsampleMetadata.subsamplingGrid, 3.f));

    // Prepare data strucutres for the subsampling
    auto pointCountGrid  = createGpuU32 (nodesBaseLevel, "pointCountGrid");
    auto averagingGrid   = createGpuAveraging (nodesBaseLevel, "averagingGrid");
    auto denseToSpareLUT = createGpuI32 (nodesBaseLevel, "denseToSpareLUT");
    auto voxelCount      = createGpuU32 (1, "voxelCount");

    pointCountGrid->memset (0);
    denseToSpareLUT->memset (-1);
    voxelCount->memset (0);

    SubsamplingTimings timings = {};

    auto randomStates = createGpuRandom (1024, "randomStates");

    // ToDo: Time measurement
    executeKernel (subsampling::kernelInitRandoms, 1024, std::time (0), randomStates->devicePointer (), 1024);
    auto randomIndices = createGpuU32 (nodesBaseLevel, "randomIndices");

    timings = randomSubsampling (
            h_sparseToDenseLUT,
            getRootIndex (),
            itsMetadata.depth,
            pointCountGrid,
            averagingGrid,
            denseToSpareLUT,
            voxelCount,
            randomStates,
            randomIndices);


    itsTimeMeasurement.emplace_back ("subsampleEvaluation", timings.subsampleEvaluation);
    itsTimeMeasurement.emplace_back ("generateRandoms", timings.generateRandoms);
    itsTimeMeasurement.emplace_back ("subsampling", timings.subsampling);
    spdlog::info ("[kernel] kernelEvaluateSubsamples took: {}[ms]", timings.subsampleEvaluation);
    spdlog::info ("[kernel] kernelGenerateRandoms took: {}[ms]", timings.generateRandoms);
    spdlog::info ("[kernel] kernelRandomPointSubsample took: {}[ms]", timings.subsampling);
}


SubsamplingTimings OctreeProcessor::OctreeProcessorImpl::randomSubsampling (
        const unique_ptr<int[]>& h_sparseToDenseLUT,
        uint32_t sparseVoxelIndex,
        uint32_t level,
        GpuArrayU32& subsampleCountingGrid,
        GpuAveraging& averagingGrid,
        GpuArrayI32& subsampleDenseToSparseLUT,
        GpuArrayU32& subsampleSparseVoxelCount,
        GpuRandomState& randomStates,
        GpuArrayU32& randomIndices)
{
    SubsamplingTimings timings = {};

    auto& cloudMetadata = itsCloud->getMetadata ();
    auto& node          = itsOctreeData->getNode (sparseVoxelIndex);

    // Depth first traversal
    for (int childIndex : node.childrenChunks)
    {
        if (childIndex != -1)
        {
            SubsamplingTimings childTiming = randomSubsampling (
                    h_sparseToDenseLUT,
                    childIndex,
                    level - 1,
                    subsampleCountingGrid,
                    averagingGrid,
                    subsampleDenseToSparseLUT,
                    subsampleSparseVoxelCount,
                    randomStates,
                    randomIndices);

            timings.subsampleEvaluation += childTiming.subsampleEvaluation;
            timings.generateRandoms += childTiming.generateRandoms;
            timings.subsampling += childTiming.subsampling;
        }
    }

    // Now we can assure that all direct children have subsamples
    if (node.isParent)
    {
        // Prepare and update the SubsampleConfig on the GPU
        SubsampleSet subsampleSet{};
        uint32_t maxPoints = prepareSubsampleConfig (subsampleSet, sparseVoxelIndex);

        // Parent bounding box calculation
        PointCloudMetadata metadata = cloudMetadata;
        auto denseVoxelIndex        = h_sparseToDenseLUT[sparseVoxelIndex];
        calculateVoxelBB (metadata, denseVoxelIndex, level);

        Kernel::KernelConfig kernelConfig = {metadata.cloudType, maxPoints};
        KernelStructs::Cloud cloud        = {itsCloud->getCloudDevice (), 0, metadata.pointDataStride};
        KernelStructs::Gridding gridding  = {
                itsSubsampleMetadata.subsamplingGrid, metadata.cubicSize (), metadata.bbCubic.min};

        // Evaluate how many points fall in each cell
        timings.subsampleEvaluation += Kernel::evaluateSubsamples (
                kernelConfig,
                subsampleSet,
                subsampleCountingGrid->devicePointer (),
                averagingGrid->devicePointer (),
                subsampleDenseToSparseLUT->devicePointer (),
                subsampleSparseVoxelCount->devicePointer (),
                cloud,
                gridding);

        // Prepare one random point index per cell
        uint32_t threads = subsampleDenseToSparseLUT->pointCount ();
        timings.generateRandoms += executeKernel (
                subsampling::kernelGenerateRandoms,
                threads,
                randomStates->devicePointer (),
                randomIndices->devicePointer (),
                subsampleDenseToSparseLUT->devicePointer (),
                subsampleCountingGrid->devicePointer (),
                threads);

        // Reserve memory for a data LUT for the parent node
        auto amountUsedVoxels = subsampleSparseVoxelCount->toHost ()[0];
        // Create LUT and averaging data for parent node
        auto subsampleLUT  = createGpuU32 (amountUsedVoxels, "subsampleLUT_" + to_string (sparseVoxelIndex));
        auto averagingData = createGpuAveraging (amountUsedVoxels, "averagingData_" + to_string (sparseVoxelIndex));
        averagingData->memset (0);
        itsParentLut.insert (make_pair (sparseVoxelIndex, move (subsampleLUT)));
        itsAveragingData.insert (make_pair (sparseVoxelIndex, move (averagingData)));

        // Distribute the subsampled points in parallel for all child nodes
        timings.subsampling += Kernel::randomPointSubsampling (
                kernelConfig,
                subsampleSet,
                itsParentLut[sparseVoxelIndex]->devicePointer (),
                itsAveragingData[sparseVoxelIndex]->devicePointer (),
                subsampleCountingGrid->devicePointer (),
                averagingGrid->devicePointer (),
                subsampleDenseToSparseLUT->devicePointer (),
                subsampleSparseVoxelCount->devicePointer (),
                cloud,
                gridding,
                randomIndices->devicePointer (),
                itsSubsampleMetadata.useReplacementScheme);
    }

    return timings;
}


uint32_t OctreeProcessor::OctreeProcessorImpl::prepareSubsampleConfig (SubsampleSet& subsampleSet, uint32_t parentIndex)
{
    uint32_t maxPoints = 0;
    auto* config       = (SubsampleConfig*)(&subsampleSet);
    auto& node         = itsOctreeData->getNode (parentIndex);
    for (uint8_t i = 0; i < 8; ++i)
    {
        int childIndex = node.childrenChunks[i];
        if (childIndex != -1)
        {
            Chunk child               = itsOctreeData->getNode (childIndex);
            config[i].pointAmount     = child.isParent ? itsParentLut[childIndex]->pointCount () : child.pointCount;
            maxPoints                 = max (maxPoints, config[i].pointAmount);
            config[i].averagingAdress = child.isParent ? itsAveragingData[childIndex]->devicePointer () : nullptr;
            config[i].lutStartIndex   = child.isParent ? 0 : child.chunkDataIndex;
            config[i].lutAdress =
                    child.isParent ? itsParentLut[childIndex]->devicePointer () : itsLeafLut->devicePointer ();
        }
        else
        {
            config[i].pointAmount     = 0;
            config[i].averagingAdress = nullptr;
            config[i].lutAdress       = nullptr;
        }
    }
    return maxPoints;
}
