#include "kernel_executor.cuh"
#include "octree_processor_impl.cuh"
#include "random_subsampling.cuh"
#include "subsample_evaluating.cuh"
#include "time_tracker.cuh"


void OctreeProcessor::OctreeProcessorImpl::performSubsampling ()
{
    auto h_octreeSparse = itsOctreeData->getHost ();
    auto h_sparseToDenseLUT = itsSparseToDenseLUT->toHost ();

    uint32_t pointSum = 0;
    evaluateOctreeProperties (
            h_octreeSparse,
            itsMetadata.leafNodeAmount,
            itsMetadata.parentNodeAmount,
            pointSum,
            itsMetadata.minPointsPerNode,
            itsMetadata.maxPointsPerNode,
            getRootIndex ());

    itsSubsamples->configureNodeAmount (itsMetadata.leafNodeAmount + itsMetadata.parentNodeAmount);

    auto start = std::chrono::high_resolution_clock::now ();
    SubsamplingTimings timings = randomSubsampling (h_sparseToDenseLUT, getRootIndex (), itsMetadata.depth);
    auto finish                           = std::chrono::high_resolution_clock::now ();
    std::chrono::duration<double> elapsed = finish - start;
    spdlog::error("randomSubsampling took: {} s", elapsed.count());

    auto& tracker = TimeTracker::getInstance ();
    tracker.trackKernelTime (timings.offsetCalcuation, "kernelCalcNodeByteOffset");
    tracker.trackKernelTime (timings.subsampleEvaluation, "kernelEvaluateSubsamples");
    tracker.trackKernelTime (timings.generateRandoms, "kernelGenerateRandoms");
    tracker.trackKernelTime (timings.subsampling, "kernelRandomPointSubsample");
}


SubsamplingTimings OctreeProcessor::OctreeProcessorImpl::randomSubsampling (
        const unique_ptr<int[]>& h_sparseToDenseLUT, uint32_t sparseVoxelIndex, uint32_t level)
{
    SubsamplingTimings timings = {};

    auto& cloudMetadata = itsCloud->getMetadata ();
    auto& node          = itsOctreeData->getNode (sparseVoxelIndex);

    // Depth first traversal
    for (int childIndex : node.childrenChunks)
    {
        if (childIndex != -1)
        {
            SubsamplingTimings childTiming = randomSubsampling (h_sparseToDenseLUT, childIndex, level - 1);

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
        prepareSubsampleConfig (subsampleSet, sparseVoxelIndex);

        // Parent bounding box calculation
        PointCloudMetadata metadata = cloudMetadata;
        auto denseVoxelIndex        = h_sparseToDenseLUT[sparseVoxelIndex];
        calculateVoxelBB (metadata, denseVoxelIndex, level);

        // ToDo: Find more sprecise amount of threads
        Kernel::KernelConfig kernelConfig = {metadata.cloudType, itsMetadata.maxPointsPerNode * 8};
        KernelStructs::Cloud cloud        = {itsCloud->getCloudDevice (), 0, metadata.pointDataStride};
        KernelStructs::Gridding gridding  = {
                itsSubsampleMetadata.subsamplingGrid, metadata.cubicSize (), metadata.bbCubic.min};

        timings.offsetCalcuation +=
                Kernel::calcNodeByteOffset ({metadata.cloudType, 1}, itsSubsamples->getNodeOutputDevice (), linearIdx);

        // Evaluate how many points fall in each cell
        timings.subsampleEvaluation += Kernel::evaluateSubsamples (
                kernelConfig,
                subsampleSet,
                itsSubsamples->getCountingGrid_d (),
                itsSubsamples->getAverageingGrid_d (),
                itsSubsamples->getDenseToSparseLut_d (),
                itsSubsamples->getOutputDevice (),
                itsSubsamples->getNodeOutputDevice (),
                linearIdx,
                cloud,
                gridding,
                itsLeafLut->devicePointer ());

        // Prepare one random point index per cell
        auto threads = itsSubsamples->getGridCellAmount();

        timings.generateRandoms += executeKernel (
                subsampling::kernelGenerateRandoms,
                threads,
                itsSubsamples->getRandomStates_d (),
                itsSubsamples->getRandomIndices_d (),
                itsSubsamples->getDenseToSparseLut_d (),
                itsSubsamples->getCountingGrid_d (),
                threads);

        // Distribute the subsampled points in parallel for all child nodes
        timings.subsampling += Kernel::randomPointSubsampling (
                kernelConfig,
                subsampleSet,
                itsSubsamples->getCountingGrid_d (),
                itsSubsamples->getAverageingGrid_d (),
                itsSubsamples->getDenseToSparseLut_d (),
                cloud,
                gridding,
                itsSubsamples->getRandomIndices_d (),
                itsSubsamples->getOutputDevice (),
                itsSubsamples->getNodeOutputDevice (),
                linearIdx,
                itsLeafLut->devicePointer ());
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
            config[i].leafPointAmount = child.pointCount;
            config[i].leafDataIdx     = child.chunkDataIndex;
            // config[i].averagingAdress = child.isParent ? itsSubsamples->getAvgDevice (childIndex) : nullptr;
            // config[i].lutStartIndex   = child.isParent ? 0 : child.chunkDataIndex;
            // config[i].lutAdress =
            //        child.isParent ? itsSubsamples->getLutDevice (childIndex) : itsLeafLut->devicePointer ();
        }
        else
        {
            // config[i].averagingAdress = nullptr;
            // config[i].lutAdress       = nullptr;
            config[i].isParent = false;
        }
    }
}
