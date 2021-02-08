#include "kernel_executor.cuh"
#include "octree_processor.h"
#include "random_subsampling.cuh"
#include "subsample_evaluating.cuh"


float OctreeProcessor::initRandomStates (
        unsigned int seed, GpuRandomState& states, uint32_t nodeAmount)
{
    return executeKernel (subsampling::kernelInitRandoms, nodeAmount, seed, states->devicePointer (), nodeAmount);
}



SubsamplingTimings OctreeProcessor::randomSubsampling (
        const unique_ptr<Chunk[]>& h_octreeSparse,
        const unique_ptr<int[]>& h_sparseToDenseLUT,
        uint32_t sparseVoxelIndex,
        uint32_t level,
        GpuArrayU32& subsampleCountingGrid,
        GpuArrayI32& subsampleDenseToSparseLUT,
        GpuArrayU32& subsampleSparseVoxelCount,
        GpuRandomState& randomStates,
        GpuArrayU32& randomIndices)
{
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
        uint32_t accumulatedPoints = 0;
        SubsampleSet subsampleSet {};
        prepareSubsampleConfig (subsampleSet, voxel, h_octreeSparse, accumulatedPoints);

        // Parent bounding box calculation
        PointCloudMetadata metadata = itsMetadata.cloudMetadata;
        auto denseVoxelIndex                        = h_sparseToDenseLUT[sparseVoxelIndex];
        calculateVoxelBB (metadata, denseVoxelIndex, level);

        // Evaluate the subsample points in parallel for all child nodes
        timings.subsampleEvaluation += Kernel::evaluateSubsamples(
                {
                    metadata.cloudType,
                    accumulatedPoints
                },
                itsCloud->getCloudDevice(),
                subsampleSet,
                subsampleCountingGrid->devicePointer (),
                subsampleDenseToSparseLUT->devicePointer (),
                subsampleSparseVoxelCount->devicePointer (),
                metadata,
                itsMetadata.subsamplingGrid,
                accumulatedPoints);


        // Reserve memory for a data LUT for the parent node
        auto amountUsedVoxels = subsampleSparseVoxelCount->toHost ()[0];

        // Create LUT and averaging data for parent node
        auto subsampleLUT  = createGpuU32 (amountUsedVoxels, "subsampleLUT_" + to_string (sparseVoxelIndex));
        auto averagingData = createGpuAveraging (amountUsedVoxels, "averagingData_" + to_string (sparseVoxelIndex));
        itsParentLut.insert (make_pair (sparseVoxelIndex, move (subsampleLUT)));
        itsAveragingData.insert (make_pair (sparseVoxelIndex, move (averagingData)));

        // Prepare random point indices and reset averaging data
        uint32_t threads = subsampleDenseToSparseLUT->pointCount ();
        timings.generateRandoms += executeKernel (
                subsampling::kernelGenerateRandoms,
                threads,
                randomStates->devicePointer (),
                randomIndices->devicePointer (),
                subsampleDenseToSparseLUT->devicePointer (),
                itsAveragingData[sparseVoxelIndex]->devicePointer (),
                subsampleCountingGrid->devicePointer (),
                threads);

        // Perform averaging in parallel for all child nodes
        timings.averaging += Kernel::performAveraging (
                {
                  metadata.cloudType,
                  accumulatedPoints
                },
                itsCloud->getCloudDevice(),
                subsampleSet,
                itsAveragingData[sparseVoxelIndex]->devicePointer (),
                subsampleDenseToSparseLUT->devicePointer (),
                metadata,
                itsMetadata.subsamplingGrid,
                accumulatedPoints);

        // Distribute the subsampled points in parallel for all child nodes
        timings.subsampling += Kernel::randomPointSubsampling (
                {
                  metadata.cloudType,
                  accumulatedPoints
                },
                itsCloud->getCloudDevice(),
                subsampleSet,
                itsParentLut[sparseVoxelIndex]->devicePointer (),
                subsampleCountingGrid->devicePointer (),
                subsampleDenseToSparseLUT->devicePointer (),
                subsampleSparseVoxelCount->devicePointer (),
                metadata,
                itsMetadata.subsamplingGrid,
                randomIndices->devicePointer (),
                accumulatedPoints);
    }

    return timings;
}
