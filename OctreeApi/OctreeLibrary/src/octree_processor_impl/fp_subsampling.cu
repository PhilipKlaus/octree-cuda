#include "fp_subsampling.cuh"
#include "fp_subsampling_evaluation.cuh"
#include "kernel_executor.cuh"
#include "kernel_helpers.cuh"
#include "kernel_structs.cuh"
#include "octree_processor_impl.cuh"

void OctreeProcessor::OctreeProcessorImpl::firstPointSubsampling (
        const unique_ptr<int[]>& h_sparseToDenseLUT, uint32_t sparseVoxelIndex, uint32_t level)
{
    auto& cloudMetadata = itsCloud->getMetadata ();
    auto& node          = itsOctree->getNode (sparseVoxelIndex);

    // Depth first traversal
    for (int childIndex : node.childNodes)
    {
        if (childIndex != -1)
        {
            firstPointSubsampling (h_sparseToDenseLUT, childIndex, level - 1);
        }
    }

    // Now we can assure that all direct children have subsamples
    if (node.isParent)
    {
        // Parent bounding box calculation
        PointCloudInfo metadata = cloudMetadata;
        auto denseVoxelIndex    = h_sparseToDenseLUT[sparseVoxelIndex];
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
                itsProcessingInfo.subsamplingGrid, metadata.cubicSize (), metadata.bbCubic.min};

        Kernel::calcNodeByteOffset (
                {metadata.cloudType, 1, "kernelCalcNodeByteOffset"},
                itsOctree->getDevice (),
                sparseVoxelIndex,
                getLastParent (),
                itsTmpCounting->devicePointer ());

        setActiveParent (sparseVoxelIndex);

        // No color averagin is performed -> directly subsample points
        if (!itsProcessingInfo.useIntraCellAvg && !itsProcessingInfo.useInterCellAvg)
        {
            Kernel::fp::firstPointSubsamplingNotAveraged (
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
                    sparseVoxelIndex);

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
                        sparseVoxelIndex);
            }
            // Inter-cell averaging
            else
            {
                Kernel::fp::evaluateSubsamplesInter (
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
            Kernel::fp::firstPointSubsampling (
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
    }
}
