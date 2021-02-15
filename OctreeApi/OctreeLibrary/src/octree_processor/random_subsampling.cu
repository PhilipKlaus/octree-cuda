#include "kernel_executor.cuh"
#include "octree_processor.h"
#include "random_subsampling.cuh"
#include "subsample_evaluating.cuh"


float OctreeProcessor::initRandomStates (unsigned int seed, GpuRandomState& states, uint32_t nodeAmount)
{
    return executeKernel (subsampling::kernelInitRandoms, nodeAmount, seed, states->devicePointer (), nodeAmount);
}


SubsamplingTimings OctreeProcessor::randomSubsampling (
        const unique_ptr<Chunk[]>& h_octreeSparse,
        const unique_ptr<int[]>& h_sparseToDenseLUT,
        uint32_t sparseVoxelIndex,
        uint32_t level,
        GpuArrayU32& subsampleCountingGrid,
        GpuAveraging & averagingGrid,
        GpuArrayI32& subsampleDenseToSparseLUT,
        GpuArrayU32& subsampleSparseVoxelCount,
        GpuRandomState& randomStates,
        GpuArrayU32& randomIndices)
{
    PointCloudMetadata cloudMetadata = itsMetadata.cloudMetadata;

    Chunk voxel                = h_octreeSparse[sparseVoxelIndex];
    SubsamplingTimings timings = {};

    // Depth first traversal
    for (int childIndex : voxel.childrenChunks)
    {
        if (childIndex != -1)
        {
            SubsamplingTimings childTiming = randomSubsampling (
                    h_octreeSparse,
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
            timings.averaging += childTiming.averaging;
            timings.subsampling += childTiming.subsampling;
        }
    }

    // Now we can assure that all direct children have subsamples
    if (voxel.isParent)
    {
        // Prepare and update the SubsampleConfig on the GPU
        SubsampleSet subsampleSet{};
        uint32_t maxPoints = prepareSubsampleConfig (subsampleSet, voxel, h_octreeSparse);

        // Parent bounding box calculation
        PointCloudMetadata metadata = cloudMetadata;
        auto denseVoxelIndex        = h_sparseToDenseLUT[sparseVoxelIndex];
        calculateVoxelBB (metadata, denseVoxelIndex, level);

        Kernel::KernelConfig kernelConfig      = {metadata.cloudType, maxPoints};
        KernelStructs::Cloud cloud       = {itsCloud->getCloudDevice (), 0, metadata.pointDataStride};
        KernelStructs::Gridding gridding = {itsSubsampleMetadata.subsamplingGrid, metadata.cubicSize (), metadata.bbCubic.min};

        // Evaluate how many points fall in each cell
        timings.subsampleEvaluation += Kernel::evaluateSubsamples (
                kernelConfig,
                subsampleSet,
                subsampleCountingGrid->devicePointer (),
                averagingGrid->devicePointer(),
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
                subsampleSparseVoxelCount->devicePointer (),
                subsampleCountingGrid->devicePointer (),
                threads);

        // Reserve memory for a data LUT for the parent node
        auto amountUsedVoxels = subsampleSparseVoxelCount->toHost ()[0];
        // Create LUT and averaging data for parent node
        auto subsampleLUT  = createGpuU32 (amountUsedVoxels, "subsampleLUT_" + to_string (sparseVoxelIndex));
        auto averagingData = createGpuAveraging (amountUsedVoxels, "averagingData_" + to_string (sparseVoxelIndex));
        averagingData->memset(0);
        itsParentLut.insert (make_pair (sparseVoxelIndex, move (subsampleLUT)));
        itsAveragingData.insert (make_pair (sparseVoxelIndex, move (averagingData)));

        // Distribute the subsampled points in parallel for all child nodes
        timings.subsampling += Kernel::randomPointSubsampling (
                kernelConfig,
                subsampleSet,
                itsParentLut[sparseVoxelIndex]->devicePointer (),
                itsAveragingData[sparseVoxelIndex]->devicePointer (),
                subsampleCountingGrid->devicePointer (),
                averagingGrid->devicePointer(),
                subsampleDenseToSparseLUT->devicePointer (),
                subsampleSparseVoxelCount->devicePointer (),
                cloud,
                gridding,
                randomIndices->devicePointer (),
                itsSubsampleMetadata.useReplacementScheme);
    }

    return timings;
}
