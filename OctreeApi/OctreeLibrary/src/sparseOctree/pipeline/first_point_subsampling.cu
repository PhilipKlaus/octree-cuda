#include <sparseOctree.h>
#include <subsample_evaluating.cuh>
#include <first_point_subsampling.cuh>


template <typename coordinateType, typename colorType>
std::tuple<float, float> SparseOctree<coordinateType, colorType>::firstPointSubsampling(
        const unique_ptr<Chunk[]> &h_octreeSparse,
        const unique_ptr<int[]> &h_sparseToDenseLUT,
        uint32_t sparseVoxelIndex,
        uint32_t level,
        unique_ptr<CudaArray<uint32_t>> &subsampleCountingGrid,
        unique_ptr<CudaArray<int>> &subsampleDenseToSparseLUT,
        unique_ptr<CudaArray<uint32_t>> &subsampleSparseVoxelCount,
        unique_ptr<CudaArray<SubsampleConfig>> &subsampleConfig) {

    Chunk voxel = h_octreeSparse[sparseVoxelIndex];
    std::tuple<float, float> accumulatedTime = {0,0};

    // Depth first traversal
    for(int childIndex : voxel.childrenChunks) {
        if(childIndex != -1) {
            std::tuple<float, float> childTime = firstPointSubsampling(
                    h_octreeSparse,
                    h_sparseToDenseLUT,
                    childIndex,
                    level - 1,
                    subsampleCountingGrid,
                    subsampleDenseToSparseLUT,
                    subsampleSparseVoxelCount,
                    subsampleConfig);

            get<0>(accumulatedTime) += get<0>(childTime);
            get<1>(accumulatedTime) += get<1>(childTime);
        }
    }

    // Now we can assure that all direct children have subsamples
    if(voxel.isParent) {

        // Prepare and update the SubsampleConfig on the GPU
        uint32_t accumulatedPoints = 0;
        prepareSubsampleConfig(voxel, h_octreeSparse, subsampleConfig, accumulatedPoints);

        // Parent bounding box calculation
        PointCloudMetadata metadata = itsMetadata.cloudMetadata;
        auto denseVoxelIndex = h_sparseToDenseLUT[sparseVoxelIndex];
        calculateVoxelBB(metadata, denseVoxelIndex, level);

        // Evaluate the subsample points in parallel for all child nodes
        get<0>(accumulatedTime) += subsampling::evaluateSubsamples<float>(
                itsCloudData,
                subsampleConfig,
                subsampleCountingGrid,
                subsampleDenseToSparseLUT,
                subsampleSparseVoxelCount,
                metadata,
                itsMetadata.subsamplingGrid,
                accumulatedPoints);

        // Reserve memory for a data LUT for the parent node
        auto amountUsedVoxels = subsampleSparseVoxelCount->toHost()[0];

        auto subsampleLUT = make_unique<CudaArray<uint32_t >>(amountUsedVoxels, "subsampleLUT_" + to_string(sparseVoxelIndex));
        itsSubsampleLUTs.insert(make_pair(sparseVoxelIndex, move(subsampleLUT)));

        // Distribute the subsampled points in parallel for all child nodes
        get<1>(accumulatedTime) += subsampling::firstPointSubsample<float>(
                itsCloudData,
                subsampleConfig,
                itsSubsampleLUTs[sparseVoxelIndex],
                subsampleCountingGrid,
                subsampleDenseToSparseLUT,
                subsampleSparseVoxelCount,
                metadata,
                itsMetadata.subsamplingGrid,
                accumulatedPoints);
    }
    return accumulatedTime;
}

template std::tuple<float, float> SparseOctree<float, uint8_t>::firstPointSubsampling(const unique_ptr<Chunk[]> &h_octreeSparse,
                                                                                      const unique_ptr<int[]> &h_sparseToDenseLUT,
                                                                                      uint32_t sparseVoxelIndex,
                                                                                      uint32_t level,
                                                                                      unique_ptr<CudaArray<uint32_t>> &subsampleCountingGrid,
                                                                                      unique_ptr<CudaArray<int>> &subsampleDenseToSparseLUT,
                                                                                      unique_ptr<CudaArray<uint32_t>> &subsampleSparseVoxelCount,
                                                                                      unique_ptr<CudaArray<SubsampleConfig>> &subsampleConfig);