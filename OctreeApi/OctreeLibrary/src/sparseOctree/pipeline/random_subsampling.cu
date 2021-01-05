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
        unique_ptr<CudaArray<uint32_t>> &subsampleSparseVoxelCount) {

    Chunk voxel = h_octreeSparse[sparseVoxelIndex];
    std::tuple<float, float> accumulatedTime = {0,0};

    // 1. Depth first traversal
    for(int childIndex : voxel.childrenChunks) {
        if(childIndex != -1) {
            std::tuple<float, float> childTime = randomSubsampling(
                    h_octreeSparse,
                    h_sparseToDenseLUT,
                    childIndex,
                    level - 1,
                    subsampleCountingGrid,
                    subsampleDenseToSparseLUT,
                    subsampleSparseVoxelCount);

            get<0>(accumulatedTime) += get<0>(childTime);
            get<1>(accumulatedTime) += get<1>(childTime);
        }
    }

    // 2. Now we can assure that all direct children have subsamples
    if(voxel.isParent) {

        // 3. Calculate the dense coordinates of the voxel
        BoundingBox bb{};
        CoordinateVector<uint32_t> coords{};
        auto denseVoxelIndex = h_sparseToDenseLUT[sparseVoxelIndex];
        calculateVoxelBB(bb, coords, denseVoxelIndex, level);

        PointCloudMetadata metadata = itsMetadata.cloudMetadata;
        metadata.boundingBox = bb;
        metadata.cloudOffset = bb.minimum;

        // 4. Pre-calculate the subsamples and count the subsampled points
        for(int childIndex : voxel.childrenChunks) {

            if(childIndex != -1) {
                Chunk child = h_octreeSparse[childIndex];
                metadata.pointAmount = child.isParent ? itsSubsampleLUTs[childIndex]->pointCount() : child.pointCount;

                get<0>(accumulatedTime) += subsampling::evaluateSubsamples<float>(
                        itsCloudData,
                        child.isParent ? itsSubsampleLUTs[childIndex] : itsDataLUT,
                        child.isParent ? 0 : child.chunkDataIndex,
                        subsampleCountingGrid,
                        subsampleDenseToSparseLUT,
                        subsampleSparseVoxelCount,
                        metadata,
                        itsMetadata.subsamplingGrid);
            }
        }

        // 5. Reserve memory for a data LUT for the parent node
        auto amountUsedVoxels = subsampleSparseVoxelCount->toHost()[0];

        //---------- GENERATE RANDOM INDICES FOR SUBSAMPLING ----------------
        auto randomStates = make_unique<CudaArray<curandState_t >>(amountUsedVoxels, "randomStates_" + to_string(sparseVoxelIndex));
        auto randomIndices = make_unique<CudaArray<uint32_t >>(amountUsedVoxels, "randomIndices_" + to_string(sparseVoxelIndex));

        // ToDo: Pre-calculate 1024 random states
        get<1>(accumulatedTime) += subsampling::initRandoms(time(0), randomStates, amountUsedVoxels);
        get<1>(accumulatedTime) += subsampling::generateRandoms(randomStates, randomIndices, subsampleDenseToSparseLUT, subsampleCountingGrid, subsampleDenseToSparseLUT->pointCount());

        //-------------------------------------------------------------------

        auto subsampleLUT = make_unique<CudaArray<uint32_t >>(amountUsedVoxels, "subsampleLUT_" + to_string(sparseVoxelIndex));
        itsSubsampleLUTs.insert(make_pair(sparseVoxelIndex, move(subsampleLUT)));

        // 6. Distribute points to the parent data LUT
        for(int childIndex : voxel.childrenChunks) {

            if(childIndex != -1) {
                Chunk child = h_octreeSparse[childIndex];
                metadata.pointAmount = child.isParent ? itsSubsampleLUTs[childIndex]->pointCount() : child.pointCount;

                get<1>(accumulatedTime) += subsampling::randomPointSubsample<float>(
                        itsCloudData,
                        child.isParent ? itsSubsampleLUTs[childIndex] : itsDataLUT,
                        child.isParent ? 0 : child.chunkDataIndex,
                        itsSubsampleLUTs[sparseVoxelIndex],
                        subsampleCountingGrid,
                        subsampleDenseToSparseLUT,
                        subsampleSparseVoxelCount,
                        metadata,
                        itsMetadata.subsamplingGrid,
                        randomIndices);
            }
        }
    }
    return accumulatedTime;
}