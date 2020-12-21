#ifndef OCTREECUDA_INDEXING_KERNELS_CUH
#define OCTREECUDA_INDEXING_KERNELS_CUH

#include <cuda_runtime.h>
#include <types.h>
#include <cudaArray.h>
#include <curand.h>
#include <curand_kernel.h>

namespace pseudo__random_subsampling {
    __global__ void kernelDistributeSubsamples(
            uint8_t *cloud,
            uint32_t *childDataLUT,
            uint32_t childDataLUTStart,
            uint32_t *parentDataLUT,
            uint32_t *countingGrid,
            int *denseToSparseLUT,
            uint32_t *sparseIndexCounter,
            PointCloudMetadata metadata,
            uint32_t gridSideLength,
            uint32_t *randomIndices);

    __global__ void kernelInitRandoms(
            unsigned int seed,
            curandState_t* states,
            uint32_t nodeAmount);

    __global__ void kernelSubsample(
            uint8_t *cloud,
            uint32_t *cloudDataLUT,
            uint32_t dataLUTStartIndex,
            uint32_t *densePointCount,
            int *denseToSparseLUT,
            uint32_t *sparseIndexCounter,
            PointCloudMetadata metadata,
            uint32_t gridSideLength);

    __global__ void kernelGenerateRandoms(
            curandState_t* states,
            uint32_t *randomIndices,
            const int *denseToSparseLUT,
            const uint32_t *countingGrid,
            uint32_t gridNodes);

    float distributeSubsamples(
            unique_ptr<CudaArray<uint8_t>> &cloud,
            unique_ptr<CudaArray<uint32_t>> &childDataLUT,
            uint32_t childDataLUTStart,
            unique_ptr<CudaArray<uint32_t>> &parentDataLUT,
            unique_ptr<CudaArray<uint32_t>> &countingGrid,
            unique_ptr<CudaArray<int>> &denseToSparseLUT,
            unique_ptr<CudaArray<uint32_t>> &sparseIndexCounter,
            PointCloudMetadata metadata,
            uint32_t gridSideLength,
            unique_ptr<CudaArray<uint32_t>> &randomIndices);

    float subsample(
            unique_ptr<CudaArray<uint8_t>> &cloud,
            unique_ptr<CudaArray<uint32_t>> &cloudDataLUT,
            uint32_t dataLUTStartIndex,
            unique_ptr<CudaArray<uint32_t>> &countingGrid,
            unique_ptr<CudaArray<int>> &denseToSparseLUT,
            unique_ptr<CudaArray<uint32_t>> &sparseIndexCounter,
            PointCloudMetadata metadata,
            uint32_t gridSideLength);

    float initRandoms(
            unsigned int seed,
            unique_ptr<CudaArray<curandState_t>> &states,
            uint32_t nodeAmount);

    float generateRandoms(
            const unique_ptr<CudaArray<curandState_t>> &states,
            unique_ptr<CudaArray<uint32_t>> &randomIndices,
            const unique_ptr<CudaArray<int>> &denseToSparseLUT,
            const unique_ptr<CudaArray<uint32_t>> &countingGrid,
            uint32_t gridNodes);
}

#endif