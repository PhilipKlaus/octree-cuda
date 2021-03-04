#include "kernel_executor.cuh"
#include "octree_processor_impl.cuh"
#include "random_subsampling.cuh"
#include "subsample_evaluating.cuh"
#include "time_tracker.cuh"


void OctreeProcessor::OctreeProcessorImpl::performSubsampling ()
{
    auto h_octreeSparse = itsOctreeData->getHost ();

    uint32_t pointSum = 0;
    evaluateOctreeProperties (
            h_octreeSparse,
            itsMetadata.leafNodeAmount,
            itsMetadata.parentNodeAmount,
            pointSum,
            itsMetadata.minPointsPerNode,
            itsMetadata.maxPointsPerNode,
            getRootIndex ());

    itsSubsamples = std::make_shared<SubsamplingData> (
            itsCloud->getMetadata ().pointAmount * 2.2, itsMetadata.leafNodeAmount + itsMetadata.parentNodeAmount);
    auto h_sparseToDenseLUT = itsSparseToDenseLUT->toHost ();
    auto nodesBaseLevel     = static_cast<uint32_t> (pow (itsSubsampleMetadata.subsamplingGrid, 3.f));

    // Prepare data strucutres for the subsampling
    auto pointCountGrid  = createGpuU32 (nodesBaseLevel, "pointCountGrid");
    auto averagingGrid   = createGpuAveraging (nodesBaseLevel, "averagingGrid");
    auto denseToSpareLUT = createGpuI32 (nodesBaseLevel, "denseToSpareLUT");

    pointCountGrid->memset (0);
    denseToSpareLUT->memset (-1);

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
            randomStates,
            randomIndices);

    auto& tracker = TimeTracker::getInstance ();
    tracker.trackKernelTime (timings.subsampleEvaluation, "kernelEvaluateSubsamples");
    tracker.trackKernelTime (timings.generateRandoms, "kernelGenerateRandoms");
    tracker.trackKernelTime (timings.subsampling, "kernelRandomPointSubsample");

    itsSubsamples->copyToHost ();
}


SubsamplingTimings OctreeProcessor::OctreeProcessorImpl::randomSubsampling (
        const unique_ptr<int[]>& h_sparseToDenseLUT,
        uint32_t sparseVoxelIndex,
        uint32_t level,
        GpuArrayU32& subsampleCountingGrid,
        GpuAveraging& averagingGrid,
        GpuArrayI32& subsampleDenseToSparseLUT,
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
        auto linearIdx = itsSubsamples->addLinearLutEntry (sparseVoxelIndex);

        // Prepare and update the SubsampleConfig on the GPU
        SubsampleSet subsampleSet{};
        //spdlog::error(0);
        prepareSubsampleConfig (subsampleSet, sparseVoxelIndex);
        //spdlog::error(1);
        // Parent bounding box calculation
        PointCloudMetadata metadata = cloudMetadata;
        auto denseVoxelIndex        = h_sparseToDenseLUT[sparseVoxelIndex];
        calculateVoxelBB (metadata, denseVoxelIndex, level);

        // ToDo: Find more sprecise amount of threads
        Kernel::KernelConfig kernelConfig = {metadata.cloudType, itsMetadata.maxPointsPerNode * 8};
        KernelStructs::Cloud cloud        = {itsCloud->getCloudDevice (), 0, metadata.pointDataStride};
        KernelStructs::Gridding gridding  = {
                itsSubsampleMetadata.subsamplingGrid, metadata.cubicSize (), metadata.bbCubic.min};

        Kernel::calcNodeByteOffset ({metadata.cloudType, 1}, itsSubsamples->getNodeOutputDevice (), linearIdx);
       // spdlog::error(2);
        // Evaluate how many points fall in each cell
        timings.subsampleEvaluation += Kernel::evaluateSubsamples (
                kernelConfig,
                subsampleSet,
                subsampleCountingGrid->devicePointer (),
                averagingGrid->devicePointer (),
                subsampleDenseToSparseLUT->devicePointer (),
                itsSubsamples->getOutputDevice(),
                itsSubsamples->getNodeOutputDevice (),
                linearIdx,
                cloud,
                gridding,
                itsOctreeData->getDevice (),
                itsLeafLut->devicePointer());

        //spdlog::error(3);

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

        //spdlog::error(4);

        // Create point-LUT and averaging data
        //auto nodeOutput = itsSubsamples->getNodeOutputHost (linearIdx);
        //itsSubsamples->createLUT (nodeOutput.pointCount, sparseVoxelIndex);
        //itsSubsamples->createAvg (nodeOutput.pointCount, sparseVoxelIndex);

        // Distribute the subsampled points in parallel for all child nodes
        timings.subsampling += Kernel::randomPointSubsampling (
                kernelConfig,
                subsampleSet,
                //itsSubsamples->getLutDevice (sparseVoxelIndex),
                //itsSubsamples->getAvgDevice (sparseVoxelIndex),
                subsampleCountingGrid->devicePointer (),
                averagingGrid->devicePointer (),
                subsampleDenseToSparseLUT->devicePointer (),
                cloud,
                gridding,
                randomIndices->devicePointer (),
                itsSubsamples->getOutputDevice(),
                itsSubsamples->getNodeOutputDevice (),
                linearIdx,
                itsOctreeData->getDevice (),
                itsLeafLut->devicePointer());

        //spdlog::error(5);
    }

    return timings;
}


void OctreeProcessor::OctreeProcessorImpl::prepareSubsampleConfig (SubsampleSet& subsampleSet, uint32_t parentIndex)
{
    auto* config = (SubsampleConfig*)(&subsampleSet);
    auto& node   = itsOctreeData->getNode (parentIndex);
    for (uint8_t i = 0; i < 8; ++i)
    {
        int childIndex      = node.childrenChunks[i];
        config[i].sparseIdx = childIndex;
        if (childIndex != -1)
        {
            Chunk child               = itsOctreeData->getNode (childIndex);
            config[i].linearIdx       = itsSubsamples->getLinearIdx (childIndex); // 0 if not existing
            config[i].isParent        = child.isParent;
            //config[i].averagingAdress = child.isParent ? itsSubsamples->getAvgDevice (childIndex) : nullptr;
            //config[i].lutStartIndex   = child.isParent ? 0 : child.chunkDataIndex;
            //config[i].lutAdress =
            //        child.isParent ? itsSubsamples->getLutDevice (childIndex) : itsLeafLut->devicePointer ();
        }
        else
        {
            //config[i].averagingAdress = nullptr;
            //config[i].lutAdress       = nullptr;
            config[i].isParent        = false;
        }
    }
}
