#include <sparseOctree.h>

#include <cstdint>
#include <kernel_executor.cuh>
#include <random_subsampling.cuh>
#include <subsample_evaluating.cuh>

template <typename coordinateType, typename colorType>
float SparseOctree<coordinateType, colorType>::initRandomStates (
        unsigned int seed, unique_ptr<CudaArray<curandState_t>>& states, uint32_t nodeAmount)
{
    return executeKernel (subsampling::kernelInitRandoms, nodeAmount, seed, states->devicePointer (), nodeAmount);
}


template <typename coordinateType, typename colorType>
std::tuple<float, float> SparseOctree<coordinateType, colorType>::randomSubsampling (
        const unique_ptr<Chunk[]>& h_octreeSparse,
        const unique_ptr<int[]>& h_sparseToDenseLUT,
        uint32_t sparseVoxelIndex,
        uint32_t level,
        unique_ptr<CudaArray<uint32_t>>& subsampleCountingGrid,
        unique_ptr<CudaArray<int>>& subsampleDenseToSparseLUT,
        unique_ptr<CudaArray<uint32_t>>& subsampleSparseVoxelCount,
        unique_ptr<CudaArray<curandState_t>>& randomStates,
        unique_ptr<CudaArray<uint32_t>>& randomIndices,
        unique_ptr<CudaArray<SubsampleConfig>>& subsampleConfig)
{
    Chunk voxel                              = h_octreeSparse[sparseVoxelIndex];
    std::tuple<float, float> accumulatedTime = {0, 0};

    // Depth first traversal
    for (int childIndex : voxel.childrenChunks)
    {
        if (childIndex != -1)
        {
            std::tuple<float, float> childTime = randomSubsampling (
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

            get<0> (accumulatedTime) += get<0> (childTime);
            get<1> (accumulatedTime) += get<1> (childTime);
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
        auto denseVoxelIndex        = h_sparseToDenseLUT[sparseVoxelIndex];
        calculateVoxelBB (metadata, denseVoxelIndex, level);

        // Evaluate the subsample points in parallel for all child nodes
        get<0> (accumulatedTime) += executeKernel (
                subsampling::kernelEvaluateSubsamples<float>,
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
        get<1> (accumulatedTime) += executeKernel (
                subsampling::kernelGenerateRandoms,
                threads,
                randomStates->devicePointer (),
                randomIndices->devicePointer (),
                subsampleDenseToSparseLUT->devicePointer (),
                itsAveragingData[sparseVoxelIndex]->devicePointer (),
                subsampleCountingGrid->devicePointer (),
                threads);

        // Perform averaging in parallel for all child nodes
        get<1> (accumulatedTime) += executeKernel (
                subsampling::kernelPerformAveraging<float, uint8_t>,
                accumulatedPoints,
                itsCloudData->devicePointer (),
                subsampleConfig->devicePointer (),
                itsAveragingData[sparseVoxelIndex]->devicePointer (),
                subsampleDenseToSparseLUT->devicePointer (),
                metadata,
                itsMetadata.subsamplingGrid,
                accumulatedPoints);

        // Distribute the subsampled points in parallel for all child nodes
        get<1> (accumulatedTime) += executeKernel (
                subsampling::kernelRandomPointSubsample<float>,
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

    return accumulatedTime;
}


//----------------------------------------------------------------------------------------------------------------------
//                                           SparseOctree<float, uint8_t>
//----------------------------------------------------------------------------------------------------------------------

template float SparseOctree<float, uint8_t>::initRandomStates (
        unsigned int seed, unique_ptr<CudaArray<curandState_t>>& states, uint32_t nodeAmount);
template std::tuple<float, float> SparseOctree<float, uint8_t>::randomSubsampling (
        const unique_ptr<Chunk[]>& h_octreeSparse,
        const unique_ptr<int[]>& h_sparseToDenseLUT,
        uint32_t sparseVoxelIndex,
        uint32_t level,
        unique_ptr<CudaArray<uint32_t>>& subsampleCountingGrid,
        unique_ptr<CudaArray<int>>& subsampleDenseToSparseLUT,
        unique_ptr<CudaArray<uint32_t>>& subsampleSparseVoxelCount,
        unique_ptr<CudaArray<curandState_t>>& randomStates,
        unique_ptr<CudaArray<uint32_t>>& randomIndices,
        unique_ptr<CudaArray<SubsampleConfig>>& subsampleConfig);

//----------------------------------------------------------------------------------------------------------------------
//                                           SparseOctree<double, uint16_t>
//----------------------------------------------------------------------------------------------------------------------

template float SparseOctree<double, uint16_t>::initRandomStates (
        unsigned int seed, unique_ptr<CudaArray<curandState_t>>& states, uint32_t nodeAmount);
template std::tuple<float, float> SparseOctree<double, uint16_t>::randomSubsampling (
        const unique_ptr<Chunk[]>& h_octreeSparse,
        const unique_ptr<int[]>& h_sparseToDenseLUT,
        uint32_t sparseVoxelIndex,
        uint32_t level,
        unique_ptr<CudaArray<uint32_t>>& subsampleCountingGrid,
        unique_ptr<CudaArray<int>>& subsampleDenseToSparseLUT,
        unique_ptr<CudaArray<uint32_t>>& subsampleSparseVoxelCount,
        unique_ptr<CudaArray<curandState_t>>& randomStates,
        unique_ptr<CudaArray<uint32_t>>& randomIndices,
        unique_ptr<CudaArray<SubsampleConfig>>& subsampleConfig);