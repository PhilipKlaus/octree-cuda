#include "kernel_executor.cuh"
#include "octree_processor_impl.cuh"
#include "random_subsampling.cuh"
#include "subsample_evaluating.cuh"
#include "time_tracker.cuh"


void OctreeProcessor::OctreeProcessorImpl::performSubsampling ()
{
    auto h_sparseToDenseLUT = itsSparseToDenseLUT->toHost ();

    itsDenseToSparseLUT->memset (-1);
    itsCountingGrid->memset (0);
    auto timing = Timing::TimeTracker::start ();
    itsOctree->updateNodeStatistics ();
    Timing::TimeTracker::stop (timing, "Update octree statistics (CPU)", Timing::Time::PROCESS);
    randomSubsampling (h_sparseToDenseLUT, itsOctree->getRootIndex (), itsOctree->getMetadata ().depth);
    cudaDeviceSynchronize ();
}


void OctreeProcessor::OctreeProcessorImpl::randomSubsampling (
        const unique_ptr<int[]>& h_sparseToDenseLUT, uint32_t sparseVoxelIndex, uint32_t level)
{
    auto& cloudMetadata = itsCloud->getMetadata ();
    auto& node          = itsOctree->getNode (sparseVoxelIndex);

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
        // Parent bounding box calculation
        PointCloudMetadata metadata = cloudMetadata;
        auto denseVoxelIndex        = h_sparseToDenseLUT[sparseVoxelIndex];
        calculateVoxelBB (metadata, denseVoxelIndex, level);

        // ToDo: Find more sprecise amount of threads
        KernelStructs::Cloud cloud = {
                itsCloud->getCloudDevice (),
                0,
                metadata.pointDataStride,
                {
                        1.0 / metadata.scale.x,
                        1.0 / metadata.scale.y,
                        1.0 / metadata.scale.z,
                }};
        KernelStructs::Gridding gridding = {
                itsSubsampleMetadata.subsamplingGrid, metadata.cubicSize (), metadata.bbCubic.min};

        Kernel::calcNodeByteOffset (
                {metadata.cloudType, 1, "kernelCalcNodeByteOffset"},
                itsOctree->getDevice (),
                sparseVoxelIndex,
                getLastParent (),
                itsTmpCounting->devicePointer ());

        setActiveParent (sparseVoxelIndex);

        // Evaluate how many points fall in each cell
        Kernel::evaluateSubsamples (
                {metadata.cloudType, itsOctree->getNodeStatistics ().maxPointsPerNode * 8, "kernelEvaluateSubsamples"},
                itsCloud->getOutputBuffer_d (),
                itsCountingGrid->devicePointer (),
                itsOctree->getDevice (),
                itsAveragingGrid->devicePointer (),
                itsDenseToSparseLUT->devicePointer (),
                itsPointLut->devicePointer (),
                cloud,
                gridding,
                sparseVoxelIndex);

        // Prepare one random point index per cell
        auto threads = static_cast<uint32_t> (pow (itsSubsampleMetadata.subsamplingGrid, 3.f));

        executeKernel (
                subsampling::kernelGenerateRandoms,
                threads,
                "kernelGenerateRandoms",
                itsRandomStates->devicePointer (),
                itsRandomIndices->devicePointer (),
                itsDenseToSparseLUT->devicePointer (),
                itsCountingGrid->devicePointer (),
                threads);

        // Distribute the subsampled points in parallel for all child nodes
        Kernel::randomPointSubsampling (
                {metadata.cloudType,
                 itsOctree->getNodeStatistics ().maxPointsPerNode * 8,
                 "kernelRandomPointSubsample"},
                itsCloud->getOutputBuffer_d (),
                itsCountingGrid->devicePointer (),
                itsAveragingGrid->devicePointer (),
                itsDenseToSparseLUT->devicePointer (),
                cloud,
                gridding,
                itsRandomIndices->devicePointer (),
                itsPointLut->devicePointer (),
                itsOctree->getDevice (),
                sparseVoxelIndex);
    }
}
