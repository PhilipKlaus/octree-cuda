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
        SubsampleSet subsampleSet{};
        uint32_t maxPoints = prepareSubsampleConfig (subsampleSet, voxel, h_octreeSparse, accumulatedPoints);

        SubsampleSetTest test{};
        auto* config = (SubsampleConfigTest*)(&test);

        for (uint8_t i = 0; i < 8; ++i)
        {
            int childIndex = voxel.childrenChunks[i];
            if(childIndex != -1) {
                Chunk child = h_octreeSparse[childIndex];
                config[i].pointAmount = child.isParent ? itsParentLut[childIndex]->pointCount () : child.pointCount;
                config[i].averagingAdress  = child.isParent ? itsAveragingData[childIndex]->devicePointer () : nullptr;
                config[i].lutStartIndex    = child.isParent ? 0 : child.chunkDataIndex;
                config[i].lutAdress =
                        child.isParent ? itsParentLut[childIndex]->devicePointer () : itsLeafLut->devicePointer ();
            }
            else {
                config[i].pointAmount = 0;
                config[i].averagingAdress = nullptr;
                config[i].lutAdress = nullptr;
            }
        }

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
                test,
                subsampleCountingGrid->devicePointer (),
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


        // Perform averaging in parallel for all child nodes
        timings.averaging += Kernel::performAveraging (
                kernelConfig,
                test,
                itsAveragingData[sparseVoxelIndex]->devicePointer (),
                subsampleDenseToSparseLUT->devicePointer (),
                cloud,
                gridding);

        // Distribute the subsampled points in parallel for all child nodes
        timings.subsampling += Kernel::randomPointSubsampling (
                kernelConfig,
                test,
                itsParentLut[sparseVoxelIndex]->devicePointer (),
                subsampleCountingGrid->devicePointer (),
                subsampleDenseToSparseLUT->devicePointer (),
                subsampleSparseVoxelCount->devicePointer (),
                cloud,
                gridding,
                randomIndices->devicePointer (),
                itsSubsampleMetadata.useReplacementScheme);
    }

    return timings;
}
