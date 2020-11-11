#ifndef OCTREECUDA_INDEXING_KERNELS_CUH
#define OCTREECUDA_INDEXING_KERNELS_CUH

#include <cuda_runtime.h>
#include <types.h>
#include <cudaArray.h>

namespace indexing {
    __global__ void kernelDistributeSubsamples(
            Vector3 *cloud,
            uint32_t *childDataLUT,
            uint32_t childDataLUTStart,
            uint32_t *parentDataLUT,
            uint32_t *countingGrid,
            int *denseToSparseLUT,
            PointCloudMetadata metadata,
            uint32_t gridSideLength
    );

    __global__ void kernelSimpleSubsampling(
            Vector3 *cloud,
            uint32_t *cloudDataLUT,
            uint32_t dataLUTStartIndex,
            uint32_t *densePointCount,
            int *denseToSparseLUT,
            uint32_t *sparseIndexCounter,
            PointCloudMetadata metadata,
            uint32_t gridSideLength
    );

    float distributeSubsamples(
            unique_ptr<CudaArray<Vector3>> &cloud,
            unique_ptr<CudaArray<uint32_t>> &childDataLUT,
            uint32_t childDataLUTStart,
            unique_ptr<CudaArray<uint32_t>> &parentDataLUT,
            unique_ptr<CudaArray<uint32_t>> &countingGrid,
            unique_ptr<CudaArray<int>> &denseToSparseLUT,
            PointCloudMetadata metadata,
            uint32_t gridSideLength
    );

    float simpleSubsampling(
            unique_ptr<CudaArray<Vector3>> &cloud,
            unique_ptr<CudaArray<uint32_t>> &cloudDataLUT,
            uint32_t dataLUTStartIndex,
            unique_ptr<CudaArray<uint32_t>> &countingGrid,
            unique_ptr<CudaArray<int>> &denseToSparseLUT,
            unique_ptr<CudaArray<uint32_t>> &sparseIndexCounter,
            PointCloudMetadata metadata,
            uint32_t gridSideLength
    );
}

#endif