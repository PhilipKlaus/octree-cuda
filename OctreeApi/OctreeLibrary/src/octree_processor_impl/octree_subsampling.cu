#include "kernel_executor.cuh"
#include "octree_processor_impl.cuh"
#include "random_subsampling.cuh"
#include "subsample_evaluating.cuh"
#include "time_tracker.cuh"


void OctreeProcessor::OctreeProcessorImpl::performSubsampling ()
{
    auto h_octreeSparse     = itsOctreeData->getHost ();
    auto h_sparseToDenseLUT = itsSparseToDenseLUT->toHost ();

    itsDenseToSparseLUT->memset (-1);
    itsCountingGrid->memset(0);

    uint32_t pointSum = 0;
    evaluateOctreeProperties (
            h_octreeSparse,
            itsMetadata.leafNodeAmount,
            itsMetadata.parentNodeAmount,
            pointSum,
            itsMetadata.minPointsPerNode,
            itsMetadata.maxPointsPerNode,
            getRootIndex ());

    randomSubsampling (h_sparseToDenseLUT, getRootIndex (), itsMetadata.depth);
    cudaDeviceSynchronize ();
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
        // Prepare and update the SubsampleConfig on the GPU
        SubsampleSet subsampleSet{};
        prepareSubsampleConfig (subsampleSet, sparseVoxelIndex);

        // Parent bounding box calculation
        PointCloudMetadata metadata = cloudMetadata;
        auto denseVoxelIndex        = h_sparseToDenseLUT[sparseVoxelIndex];
        calculateVoxelBB (metadata, denseVoxelIndex, level);

        // ToDo: Find more sprecise amount of threads
        KernelStructs::Cloud cloud       = {itsCloud->getCloudDevice (), 0, metadata.pointDataStride, {
                1.0 / metadata.scale.x,
                1.0 / metadata.scale.y,
                1.0 / metadata.scale.z,
        }};
        KernelStructs::Gridding gridding = {
                itsSubsampleMetadata.subsamplingGrid, metadata.cubicSize (), metadata.bbCubic.min};

        Kernel::calcNodeByteOffset (
                {metadata.cloudType, 1, "kernelCalcNodeByteOffset"},
                itsOctreeData->getDevice(),
                sparseVoxelIndex,
                itsSubsamples->getLastParent(),
                itsTmpCounting->devicePointer());

        itsSubsamples->setActiveParent(sparseVoxelIndex);

        // Evaluate how many points fall in each cell
        Kernel::evaluateSubsamples (
                {metadata.cloudType, itsMetadata.maxPointsPerNode * 8, "kernelEvaluateSubsamples"},
                subsampleSet,
                itsCountingGrid->devicePointer(),
                //itsSubsamples->getCountingGrid_d (),
                itsOctreeData->getDevice(),
                itsSubsamples->getAverageingGrid_d (),
                itsDenseToSparseLUT->devicePointer(),
                itsSubsamples->getOutputDevice (),
                cloud,
                gridding,
                sparseVoxelIndex);

        // Prepare one random point index per cell
        auto threads = itsSubsamples->getGridCellAmount ();

        executeKernel (
                subsampling::kernelGenerateRandoms,
                threads,
                "kernelGenerateRandoms",
                itsSubsamples->getRandomStates_d (),
                itsSubsamples->getRandomIndices_d (),
                itsDenseToSparseLUT->devicePointer(),
                //itsSubsamples->getCountingGrid_d (),
                itsCountingGrid->devicePointer(),
                threads);

        // Distribute the subsampled points in parallel for all child nodes
        Kernel::randomPointSubsampling (
                {metadata.cloudType, itsMetadata.maxPointsPerNode * 8, "kernelRandomPointSubsample"},
                itsCloud->getOutputBuffer_d(),
                subsampleSet,
                //itsSubsamples->getCountingGrid_d (),
                itsCountingGrid->devicePointer(),
                itsSubsamples->getAverageingGrid_d (),
                itsDenseToSparseLUT->devicePointer(),
                cloud,
                gridding,
                itsSubsamples->getRandomIndices_d (),
                itsSubsamples->getOutputDevice (),
                itsOctreeData->getDevice(),
                sparseVoxelIndex);
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
            config[i].isParent        = child.isParent;
            config[i].leafPointAmount = child.pointCount;
            config[i].leafDataIdx     = child.chunkDataIndex;
        }
        else
        {
            config[i].isParent = false;
        }
    }
}
