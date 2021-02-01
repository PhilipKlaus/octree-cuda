#include "first_point_subsampling.cuh"
#include "kernel_executor.cuh"
#include "sparseOctree.h"
#include "subsample_evaluating.cuh"


template <typename coordinateType, typename colorType>
SubsamplingTimings SparseOctree<coordinateType, colorType>::firstPointSubsampling (
        const unique_ptr<Chunk[]>& h_octreeSparse,
        const unique_ptr<int[]>& h_sparseToDenseLUT,
        uint32_t sparseVoxelIndex,
        uint32_t level,
        GpuArrayU32& subsampleCountingGrid,
        GpuArrayI32& subsampleDenseToSparseLUT,
        GpuArrayU32& subsampleSparseVoxelCount,
        GpuSubsample& subsampleConfig)
{
    Chunk voxel                = h_octreeSparse[sparseVoxelIndex];
    SubsamplingTimings timings = {};

    // Depth first traversal
    for (int childIndex : voxel.childrenChunks)
    {
        if (childIndex != -1)
        {
            SubsamplingTimings childTiming = firstPointSubsampling (
                    h_octreeSparse,
                    h_sparseToDenseLUT,
                    childIndex,
                    level - 1,
                    subsampleCountingGrid,
                    subsampleDenseToSparseLUT,
                    subsampleSparseVoxelCount,
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
        PointCloudMetadata metadata = itsMetadata.cloudMetadata;
        auto denseVoxelIndex                        = h_sparseToDenseLUT[sparseVoxelIndex];
        calculateVoxelBB (metadata, denseVoxelIndex, level);

        // Evaluate the subsample points in parallel for all child nodes
        timings.subsampleEvaluation += Kernel::evaluateSubsamples(
                {
                        metadata.cloudType,
                        accumulatedPoints
                },
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

        auto subsampleLUT = createGpuU32 (amountUsedVoxels, "subsampleLUT_" + to_string (sparseVoxelIndex));
        itsSubsampleLUTs.insert (make_pair (sparseVoxelIndex, move (subsampleLUT)));

        // Distribute the subsampled points in parallel for all child nodes
        timings.subsampling += Kernel::firstPointSubsampling (
                {
                  metadata.cloudType,
                  accumulatedPoints
                },
                itsCloudData->devicePointer (),
                subsampleConfig->devicePointer (),
                itsSubsampleLUTs[sparseVoxelIndex]->devicePointer (),
                subsampleCountingGrid->devicePointer (),
                subsampleDenseToSparseLUT->devicePointer (),
                subsampleSparseVoxelCount->devicePointer (),
                metadata,
                itsMetadata.subsamplingGrid,
                accumulatedPoints);
    }
    return timings;
}


//----------------------------------------------------------------------------------------------------------------------
//                                           SparseOctree<float, uint8_t>
//----------------------------------------------------------------------------------------------------------------------

template SubsamplingTimings SparseOctree<float, uint8_t>::firstPointSubsampling (
        const unique_ptr<Chunk[]>& h_octreeSparse,
        const unique_ptr<int[]>& h_sparseToDenseLUT,
        uint32_t sparseVoxelIndex,
        uint32_t level,
        GpuArrayU32& subsampleCountingGrid,
        GpuArrayI32& subsampleDenseToSparseLUT,
        GpuArrayU32& subsampleSparseVoxelCount,
        GpuSubsample& subsampleConfig);

//----------------------------------------------------------------------------------------------------------------------
//                                           SparseOctree<double, uint8_t>
//----------------------------------------------------------------------------------------------------------------------

template SubsamplingTimings SparseOctree<double, uint8_t>::firstPointSubsampling (
        const unique_ptr<Chunk[]>& h_octreeSparse,
        const unique_ptr<int[]>& h_sparseToDenseLUT,
        uint32_t sparseVoxelIndex,
        uint32_t level,
        GpuArrayU32& subsampleCountingGrid,
        GpuArrayI32& subsampleDenseToSparseLUT,
        GpuArrayU32& subsampleSparseVoxelCount,
        GpuSubsample& subsampleConfig);