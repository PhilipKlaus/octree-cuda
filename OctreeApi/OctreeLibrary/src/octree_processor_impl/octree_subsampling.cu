#include "kernel_executor.cuh"
#include "octree_processor_impl.cuh"
#include "random_subsampling.cuh"
#include "subsample_evaluating.cuh"
#include "time_tracker.cuh"


void OctreeProcessor::OctreeProcessorImpl::performSubsampling ()
{
    auto h_octreeSparse     = itsOctreeData->getHost ();
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

    randomSubsampling (h_sparseToDenseLUT, getRootIndex (), itsMetadata.depth);
    cudaDeviceSynchronize();
}


void OctreeProcessor::OctreeProcessorImpl::randomSubsampling (
        const unique_ptr<int[]>& h_sparseToDenseLUT, uint32_t sparseVoxelIndex, uint32_t level)
{
    auto& cloudMetadata = itsCloud->getMetadata ();
    auto& node          = itsOctreeData->getNode (sparseVoxelIndex);

    // Depth first traversal
    for (int childIndex : node.childrenChunks)
    {
        if (childIndex != -1)
        {
            randomSubsampling (h_sparseToDenseLUT, childIndex, level - 1);
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
        KernelStructs::Cloud cloud       = {itsCloud->getCloudDevice (), 0, metadata.pointDataStride};
        KernelStructs::Gridding gridding = {
                itsSubsampleMetadata.subsamplingGrid, metadata.cubicSize (), metadata.bbCubic.min};

        Kernel::calcNodeByteOffset (
                {metadata.cloudType, 1, "kernelCalcNodeByteOffset"}, itsSubsamples->getNodeOutputDevice (), linearIdx);

        // Evaluate how many points fall in each cell
        Kernel::evaluateSubsamples (
                {metadata.cloudType, itsMetadata.maxPointsPerNode * 8, "kernelEvaluateSubsamples"},
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
        auto threads = itsSubsamples->getGridCellAmount ();

        executeKernel (
                subsampling::kernelGenerateRandoms,
                threads,
                "kernelGenerateRandoms",
                itsSubsamples->getRandomStates_d (),
                itsSubsamples->getRandomIndices_d (),
                itsSubsamples->getDenseToSparseLut_d (),
                itsSubsamples->getCountingGrid_d (),
                threads);

        // Distribute the subsampled points in parallel for all child nodes
        Kernel::randomPointSubsampling (
                {metadata.cloudType, itsMetadata.maxPointsPerNode * 8, "kernelRandomPointSubsample"},
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
