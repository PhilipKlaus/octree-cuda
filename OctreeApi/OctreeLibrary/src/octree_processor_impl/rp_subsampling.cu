#include "kernel_executor.cuh"
#include "kernel_helpers.cuh"
#include "octree_processor_impl.cuh"
#include "rp_subsample_evaluation.cuh"
#include "rp_subsampling.cuh"
#include "time_tracker.cuh"


void OctreeProcessor::OctreeProcessorImpl::randomSubsampling (
        uint32_t sparseVoxelIndex, uint32_t level, Vector3<double> nodeBBMin)
{
    auto& cloudMetadata = itsCloud->getMetadata ();
    auto& node          = itsOctree->getNode (sparseVoxelIndex);

    // Calculate actual cell (node) side length
    auto bbDivider = pow (2, itsOctree->getNodeStatistics ().depth - level);
    double side    = (cloudMetadata.bbCubic.max.x - cloudMetadata.bbCubic.min.x) / bbDivider;

    // Depth first traversal
    uint8_t tmpIndex = 0;
    for (int childIndex : node.childNodes)
    {
        if (childIndex != -1)
        {
            auto childBBSide           = side / 2.0;
            Vector3<double> childBBMin = nodeBBMin;
            tools::calculateChildMinBB (childBBMin, nodeBBMin, tmpIndex, childBBSide);
            randomSubsampling (childIndex, level - 1, childBBMin);
        }
        ++tmpIndex;
    }

    // Now we can assure that all direct children have subsamples
    if (node.isInternal)
    {
        // Parent bounding box calculation
        PointCloudInfo metadata = cloudMetadata;
        metadata.bbCubic.min    = nodeBBMin;
        metadata.bbCubic.max.x  = metadata.bbCubic.min.x + side;
        metadata.bbCubic.max.y  = metadata.bbCubic.min.y + side;
        metadata.bbCubic.max.z  = metadata.bbCubic.min.z + side;
        metadata.cloudOffset    = metadata.bbCubic.min;

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
                itsProcessingInfo.subsamplingGrid, metadata.cubicSize (), metadata.bbCubic.min};

        Kernel::calcNodeByteOffset (
                {metadata.cloudType, 1, "kernelCalcNodeByteOffset"},
                itsOctree->getDevice (),
                sparseVoxelIndex,
                getLastParent (),
                itsTmpCounting->devicePointer ());

        setActiveParent (sparseVoxelIndex);

        // Intra-cell color averaging: evaluate subsamples and accumulate colors
        if (itsProcessingInfo.useIntraCellAvg)
        {
            Kernel::rp::evaluateSubsamplesIntra (
                    {metadata.cloudType,
                     itsOctree->getNodeStatistics ().maxPointsPerNode * 8,
                     "kernelEvaluateSubsamplesIntra"},
                    itsCloud->getOutputBuffer_d (),
                    itsCountingGrid->devicePointer (),
                    itsOctree->getDevice (),
                    itsAveragingGrid->devicePointer (),
                    itsDenseToSparseLUT->devicePointer (),
                    itsPointLut->devicePointer (),
                    cloud,
                    gridding,
                    sparseVoxelIndex);
        }
        // Inter-cell color averaging: evaluate subsamples and accumulate colors
        else if (itsProcessingInfo.useInterCellAvg)
        {
            Kernel::rp::evaluateSubsamplesInter (
                    {metadata.cloudType,
                     itsOctree->getNodeStatistics ().maxPointsPerNode * 8,
                     "kernelEvaluateSubsamplesIntra"},
                    itsCountingGrid->devicePointer (),
                    itsOctree->getDevice (),
                    itsDenseToSparseLUT->devicePointer (),
                    itsPointLut->devicePointer (),
                    cloud,
                    gridding,
                    sparseVoxelIndex);

            Kernel::rp::interCellAvg (
                    {metadata.cloudType,
                     itsOctree->getNodeStatistics ().maxPointsPerNode * 8,
                     "kernelInterCellAveraging"},
                    itsCloud->getOutputBuffer_d (),
                    itsCountingGrid->devicePointer (),
                    itsOctree->getDevice (),
                    itsAveragingGrid->devicePointer (),
                    itsDenseToSparseLUT->devicePointer (),
                    itsPointLut->devicePointer (),
                    cloud,
                    gridding,
                    sparseVoxelIndex);
        }
        // No averaging is activated -> just evaluate the subsample points
        else
        {
            Kernel::rp::evaluateSubsamplesNotAveraged (
                    {metadata.cloudType,
                     itsOctree->getNodeStatistics ().maxPointsPerNode * 8,
                     "kernelEvaluateSubsamplesIntra"},
                    itsCountingGrid->devicePointer (),
                    itsOctree->getDevice (),
                    itsDenseToSparseLUT->devicePointer (),
                    itsPointLut->devicePointer (),
                    cloud,
                    gridding,
                    sparseVoxelIndex);
        }

        // Generate one random point index per cell
        auto threads = static_cast<uint32_t> (pow (itsProcessingInfo.subsamplingGrid, 3.f));

        executeKernel (
                subsampling::rp::kernelGenerateRandoms,
                threads,
                "kernelGenerateRandoms",
                itsRandomStates->devicePointer (),
                itsRandomIndices->devicePointer (),
                itsDenseToSparseLUT->devicePointer (),
                itsCountingGrid->devicePointer (),
                threads);

        // Finally subsample the color averaged points
        if (itsProcessingInfo.useIntraCellAvg || itsProcessingInfo.useInterCellAvg)
        {
            Kernel::rp::subsampleAveraged (
                    {metadata.cloudType,
                     itsOctree->getNodeStatistics ().maxPointsPerNode * 8,
                     "kernelRandomPointSubsample"},
                    itsCloud->getOutputBuffer_d (),
                    itsCountingGrid->devicePointer (),
                    itsAveragingGrid->devicePointer (),
                    itsDenseToSparseLUT->devicePointer (),
                    cloud,
                    gridding,
                    cloudMetadata.bbCubic,
                    itsRandomIndices->devicePointer (),
                    itsPointLut->devicePointer (),
                    itsOctree->getDevice (),
                    sparseVoxelIndex);
        }

        // Finally subsample the points without any color averaging
        else
        {
            Kernel::rp::subsampleNotAveraged (
                    {metadata.cloudType,
                     itsOctree->getNodeStatistics ().maxPointsPerNode * 8,
                     "kernelRandomPointSubsample"},
                    itsCloud->getOutputBuffer_d (),
                    itsCountingGrid->devicePointer (),
                    itsDenseToSparseLUT->devicePointer (),
                    cloud,
                    gridding,
                    cloudMetadata.bbCubic,
                    itsRandomIndices->devicePointer (),
                    itsPointLut->devicePointer (),
                    itsOctree->getDevice (),
                    sparseVoxelIndex);
        }
    }
}
