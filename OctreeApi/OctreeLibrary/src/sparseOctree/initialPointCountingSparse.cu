#include <sparseOctree.h>
#include <kernels.cuh>


void SparseOctree::initialPointCounting(uint32_t initialDepth) {

    // Pre-calculate different Octree parameters
    preCalculateOctreeParameters(initialDepth);

    // Allocate the dense point count
    itsDensePointCountPerVoxel = make_unique<CudaArray<uint32_t>>(itsVoxelAmountDense, "itsDensePointCountPerVoxel");
    gpuErrchk(cudaMemset (itsDensePointCountPerVoxel->devicePointer(), 0, itsVoxelAmountDense * sizeof(uint32_t)));

    // Allocate the conversion LUT from dense to sparse
    itsDenseToSparseLUT = make_unique<CudaArray<int>>(itsVoxelAmountDense, "denseToSparseLUT");
    gpuErrchk(cudaMemset (itsDenseToSparseLUT->devicePointer(), -1, itsVoxelAmountDense * sizeof(int)));

    // Allocate the global sparseIndexCounter
    itsVoxelAmountSparse = make_unique<CudaArray<uint32_t>>(1, "sparseVoxelAmount");
    gpuErrchk(cudaMemset (itsVoxelAmountSparse->devicePointer(), 0, 1 * sizeof(uint32_t)));

    float time = kernelExecution::executeKernelMapCloudToGrid(
            itsCloudData,
            itsDensePointCountPerVoxel,
            itsDenseToSparseLUT,
            itsVoxelAmountSparse,
            itsMetadata,
            itsGridSideLengthPerLevel[0]
            );

    itsTimeMeasurement.insert(std::make_pair("initialPointCount", time));
}