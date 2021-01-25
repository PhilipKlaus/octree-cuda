#include "sparseOctree.h"
#include "kernel_executor.cuh"
#include "random_subsampling.cuh"
#include "subsample_evaluating.cuh"


template <typename coordinateType, typename colorType>
float SparseOctree<coordinateType, colorType>::initRandomStates (
        unsigned int seed, GpuRandomState& states, uint32_t nodeAmount)
{
    return executeKernel (subsampling::kernelInitRandoms, nodeAmount, seed, states->devicePointer (), nodeAmount);
}


template <typename coordinateType, typename colorType>
SubsamplingTimings SparseOctree<coordinateType, colorType>::randomSubsampling (
        const unique_ptr<Chunk[]>& h_octreeSparse,
        const unique_ptr<int[]>& h_sparseToDenseLUT,
        uint32_t sparseVoxelIndex,
        uint32_t level,
        GpuArrayU32& subsampleCountingGrid,
        GpuArrayI32& subsampleDenseToSparseLUT,
        GpuArrayU32& subsampleSparseVoxelCount,
        GpuRandomState& randomStates,
        GpuArrayU32& randomIndices,
        GpuSubsample& subsampleConfig)
{
    Chunk voxel                              = h_octreeSparse[sparseVoxelIndex];
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
                    randomIndices,
                    subsampleConfig);

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
        prepareSubsampleConfig (voxel, h_octreeSparse, subsampleConfig, accumulatedPoints);

        // Parent bounding box calculation
        PointCloudMetadata<coordinateType> metadata = itsMetadata.cloudMetadata;
        auto denseVoxelIndex        = h_sparseToDenseLUT[sparseVoxelIndex];
        calculateVoxelBB (metadata, denseVoxelIndex, level);

        // Evaluate the subsample points in parallel for all child nodes
        timings.subsampleEvaluation += executeKernel (
                subsampling::kernelEvaluateSubsamples<coordinateType>,
                accumulatedPoints,
                itsCloudData->devicePointer (),
                subsampleConfig->devicePointer (),
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
        itsSubsampleLUTs.insert (make_pair (sparseVoxelIndex, move (subsampleLUT)));
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
        timings.averaging += executeKernel (
                subsampling::kernelPerformAveraging<coordinateType, colorType>,
                accumulatedPoints,
                itsCloudData->devicePointer (),
                subsampleConfig->devicePointer (),
                itsAveragingData[sparseVoxelIndex]->devicePointer (),
                subsampleDenseToSparseLUT->devicePointer (),
                metadata,
                itsMetadata.subsamplingGrid,
                accumulatedPoints);

        // Distribute the subsampled points in parallel for all child nodes
        timings.subsampling += executeKernel (
                subsampling::kernelRandomPointSubsample<coordinateType>,
                accumulatedPoints,
                itsCloudData->devicePointer (),
                subsampleConfig->devicePointer (),
                itsSubsampleLUTs[sparseVoxelIndex]->devicePointer (),
                itsAveragingData[sparseVoxelIndex]->devicePointer (),
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


//----------------------------------------------------------------------------------------------------------------------
//                                           SparseOctree<float, uint8_t>
//----------------------------------------------------------------------------------------------------------------------

template float SparseOctree<float, uint8_t>::initRandomStates (
        unsigned int seed, GpuRandomState& states, uint32_t nodeAmount);
template SubsamplingTimings SparseOctree<float, uint8_t>::randomSubsampling (
        const unique_ptr<Chunk[]>& h_octreeSparse,
        const unique_ptr<int[]>& h_sparseToDenseLUT,
        uint32_t sparseVoxelIndex,
        uint32_t level,
        GpuArrayU32& subsampleCountingGrid,
        GpuArrayI32& subsampleDenseToSparseLUT,
        GpuArrayU32& subsampleSparseVoxelCount,
        GpuRandomState& randomStates,
        GpuArrayU32& randomIndices,
        GpuSubsample& subsampleConfig);

//----------------------------------------------------------------------------------------------------------------------
//                                           SparseOctree<double, uint8_t>
//----------------------------------------------------------------------------------------------------------------------

template float SparseOctree<double, uint8_t>::initRandomStates (
        unsigned int seed, GpuRandomState& states, uint32_t nodeAmount);
template SubsamplingTimings SparseOctree<double, uint8_t>::randomSubsampling (
        const unique_ptr<Chunk[]>& h_octreeSparse,
        const unique_ptr<int[]>& h_sparseToDenseLUT,
        uint32_t sparseVoxelIndex,
        uint32_t level,
        GpuArrayU32& subsampleCountingGrid,
        GpuArrayI32& subsampleDenseToSparseLUT,
        GpuArrayU32& subsampleSparseVoxelCount,
        GpuRandomState& randomStates,
        GpuArrayU32& randomIndices,
        GpuSubsample& subsampleConfig);