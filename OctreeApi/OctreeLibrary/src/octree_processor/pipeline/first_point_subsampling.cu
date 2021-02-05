#include "first_point_subsampling.cuh"
#include "kernel_executor.cuh"
#include "octree_processor.h"
#include "subsample_evaluating.cuh"



SubsamplingTimings OctreeProcessor::firstPointSubsampling (
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
                itsCloud->getCloudDevice (),
                subsampleSet,
                subsampleCountingGrid->devicePointer (),
                subsampleDenseToSparseLUT->devicePointer (),
                subsampleSparseVoxelCount->devicePointer (),
                metadata,
                itsMetadata.subsamplingGrid,
                accumulatedPoints);


        // Reserve memory for a data LUT for the parent node
        auto amountUsedVoxels = subsampleSparseVoxelCount->toHost ()[0];

        auto subsampleLUT = createGpuU32 (amountUsedVoxels, "subsampleLUT_" + to_string (sparseVoxelIndex));
        itsParentLut.insert (make_pair (sparseVoxelIndex, move (subsampleLUT)));

        // Distribute the subsampled points in parallel for all child nodes
        timings.subsampling += Kernel::firstPointSubsampling (
                {
                  metadata.cloudType,
                  accumulatedPoints
                },
                itsCloud->getCloudDevice(),
                subsampleConfig->devicePointer (),
                itsParentLut[sparseVoxelIndex]->devicePointer (),
                subsampleCountingGrid->devicePointer (),
                subsampleDenseToSparseLUT->devicePointer (),
                subsampleSparseVoxelCount->devicePointer (),
                metadata,
                itsMetadata.subsamplingGrid,
                accumulatedPoints);
    }
    return timings;
}
