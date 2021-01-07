#include <sparseOctree.h>

#include <subsample_evaluating.cuh>
#include <random_subsampling.cuh>

std::tuple<float, float> SparseOctree::randomSubsampling(
        const unique_ptr<Chunk[]> &h_octreeSparse,
        const unique_ptr<int[]> &h_sparseToDenseLUT,
        uint32_t sparseVoxelIndex,
        uint32_t level,
        unique_ptr<CudaArray<uint32_t>> &subsampleCountingGrid,
        unique_ptr<CudaArray<int>> &subsampleDenseToSparseLUT,
        unique_ptr<CudaArray<uint32_t>> &subsampleSparseVoxelCount,
        unique_ptr<CudaArray<curandState_t >> &randomStates,
        unique_ptr<CudaArray<uint32_t >> &randomIndices,
        unique_ptr<CudaArray<SubsampleData>> &subsampleData) {

    Chunk voxel = h_octreeSparse[sparseVoxelIndex];
    std::tuple<float, float> accumulatedTime = {0,0};

    // Depth first traversal
    for(int childIndex : voxel.childrenChunks) {
        if(childIndex != -1) {
            std::tuple<float, float> childTime = randomSubsampling(
                    h_octreeSparse,
                    h_sparseToDenseLUT,
                    childIndex,
                    level - 1,
                    subsampleCountingGrid,
                    subsampleDenseToSparseLUT,
                    subsampleSparseVoxelCount,
                    randomStates,
                    randomIndices,
                    subsampleData);

            get<0>(accumulatedTime) += get<0>(childTime);
            get<1>(accumulatedTime) += get<1>(childTime);
        }
    }

    // Now we can assure that all direct children have subsamples
    if(voxel.isParent) {

        // Prepare subsample data and copy it to the GPU
        SubsampleData newSubsampleData[8];
        uint32_t i = 0;
        uint32_t pointOffset = 0;
        for(int childIndex : voxel.childrenChunks) {

            if(childIndex != -1) {
                Chunk child = h_octreeSparse[childIndex];
                newSubsampleData[i].lutAdress = child.isParent ? itsSubsampleLUTs[childIndex]->devicePointer() : itsDataLUT->devicePointer();
                newSubsampleData[i].lutStartIndex = child.isParent ? 0 : child.chunkDataIndex;
                newSubsampleData[i].pointOffsetLower = pointOffset;
                pointOffset += child.isParent ? itsSubsampleLUTs[childIndex]->pointCount() : child.pointCount;
                newSubsampleData[i].pointOffsetUpper = pointOffset;
                ++i;
            }
        }
        subsampleData->toGPU(reinterpret_cast<uint8_t *>(newSubsampleData));

        // Parent bounding box calculation
        BoundingBox bb{};
        CoordinateVector<uint32_t> coords{};
        auto denseVoxelIndex = h_sparseToDenseLUT[sparseVoxelIndex];
        calculateVoxelBB(bb, coords, denseVoxelIndex, level);

        PointCloudMetadata metadata = itsMetadata.cloudMetadata;
        metadata.boundingBox = bb;
        metadata.cloudOffset = bb.minimum;

        // Evaluate the subsample points in parallel for all child nodes
        get<0>(accumulatedTime) += subsampling::evaluateSubsamples<float>(
                itsCloudData,
                subsampleData,
                subsampleCountingGrid,
                subsampleDenseToSparseLUT,
                subsampleSparseVoxelCount,
                metadata,
                itsMetadata.subsamplingGrid,
                pointOffset);

        // Reserve memory for a data LUT for the parent node
        auto amountUsedVoxels = subsampleSparseVoxelCount->toHost()[0];

        // Prepare random point indices
        get<1>(accumulatedTime) += subsampling::generateRandoms(
                randomStates,
                randomIndices,
                subsampleDenseToSparseLUT,
                subsampleCountingGrid,
                subsampleDenseToSparseLUT->pointCount());


        auto subsampleLUT = make_unique<CudaArray<uint32_t >>(amountUsedVoxels, "subsampleLUT_" + to_string(sparseVoxelIndex));
        itsSubsampleLUTs.insert(make_pair(sparseVoxelIndex, move(subsampleLUT)));

        // Distribute the subsampled data in parallel for all child nodes
        get<1>(accumulatedTime) += subsampling::randomPointSubsample<float>(
                itsCloudData,
                subsampleData,
                itsSubsampleLUTs[sparseVoxelIndex],
                subsampleCountingGrid,
                subsampleDenseToSparseLUT,
                subsampleSparseVoxelCount,
                metadata,
                itsMetadata.subsamplingGrid,
                randomIndices,
                pointOffset);
    }
    return accumulatedTime;
}