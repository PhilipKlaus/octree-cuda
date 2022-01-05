#include "fp_subsample_evaluation.cuh"
#include "fp_subsampling.cuh"
#include "kernel_executor.cuh"
#include "kernel_helpers.cuh"
#include "kernel_structs.cuh"
#include "octree_processor_impl.cuh"

void OctreeProcessor::OctreeProcessorImpl::firstPointSubsampling (
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
            firstPointSubsampling (childIndex, level - 1, childBBMin);
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

        // No color averagin is performed -> directly subsample points
        if (!itsProcessingInfo.useIntraCellAvg && !itsProcessingInfo.useInterCellAvg)
        {
            Kernel::fp::subsampleNotAveraged (
                    {metadata.cloudType,
                     itsOctree->getNodeStatistics ().maxPointsPerNode * 8,
                     "kernelRandomPointSubsample"},
                    itsCloud->getOutputBuffer_d (),
                    itsCountingGrid->devicePointer (),
                    itsDenseToSparseLUT->devicePointer (),
                    cloud,
                    gridding,
                    cloudMetadata.bbCubic,
                    itsPointLut->devicePointer (),
                    itsOctree->getDevice (),
                    sparseVoxelIndex,
                    getLastParent (),
                    itsTmpCounting->devicePointer ());

            auto gridCellAmount = static_cast<uint32_t> (pow (itsProcessingInfo.subsamplingGrid, 3.f));

            executeKernel (
                    tools::kernelMemset1D<uint32_t>,
                    gridCellAmount,
                    "kernelMemset1D",
                    itsCountingGrid->devicePointer (),
                    0,
                    gridCellAmount);
        }
        else
        {
            // Intra-cell averaging
            if (itsProcessingInfo.useIntraCellAvg)
            {
                Kernel::fp::evaluateSubsamplesIntra (
                        {metadata.cloudType,
                         itsOctree->getNodeStatistics ().maxPointsPerNode * 8,
                         "kernelEvaluateSubsamplesIntra"},
                        itsCloud->getOutputBuffer_d (),
                        itsOctree->getDevice (),
                        itsAveragingGrid->devicePointer (),
                        itsDenseToSparseLUT->devicePointer (),
                        itsPointLut->devicePointer (),
                        cloud,
                        gridding,
                        sparseVoxelIndex,
                        getLastParent (),
                        itsTmpCounting->devicePointer ());
            }
            // Inter-cell averaging
            else
            {
                Kernel::fp::evaluateSubsamplesInter (
                        {metadata.cloudType,
                         itsOctree->getNodeStatistics ().maxPointsPerNode * 8,
                         "kernelEvaluateSubsamplesInter"},
                        itsCountingGrid->devicePointer (),
                        itsOctree->getDevice (),
                        itsDenseToSparseLUT->devicePointer (),
                        itsPointLut->devicePointer (),
                        cloud,
                        gridding,
                        sparseVoxelIndex,
                        getLastParent (),
                        itsTmpCounting->devicePointer ());

                // Inter-Cell: Accumulate colors from neighbouring cells
                Kernel::fp::interCellAvg (
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

            // Finally subsample the points
            Kernel::fp::subsampleAveraged (
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
                    itsPointLut->devicePointer (),
                    itsOctree->getDevice (),
                    sparseVoxelIndex);
        }

        setActiveParent (sparseVoxelIndex);
    }
}
