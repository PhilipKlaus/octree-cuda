#ifndef OCTREECUDA_CHUNKING_KERNELS_CUH
#define OCTREECUDA_CHUNKING_KERNELS_CUH

#include <cuda_runtime.h>
#include <types.h>
#include <cudaArray.h>

namespace chunking {

    //------------- Point Counting ----------------------

    __global__ void kernelInitialPointCounting(
            Vector3 *cloud,
            uint32_t *densePointCount,
            int *denseToSparseLUT,
            uint32_t *sparseIndexCounter,
            PointCloudMetadata metadata,
            uint32_t gridSideLength
    );

    float initialPointCounting(
            unique_ptr<CudaArray<Vector3>> &cloud,
            unique_ptr<CudaArray<uint32_t>> &densePointCount,
            unique_ptr<CudaArray<int>> &denseToSparseLUT,
            unique_ptr<CudaArray<uint32_t>> &sparseIndexCounter,
            PointCloudMetadata metadata,
            uint32_t gridSideLength
    );

    //--------- Octree Initialization / Merging -------

    __global__ void kernelInitLowestOctreeHierarchy(
            Chunk *octreeSparse,
            uint32_t *densePointCount,
            int *denseToSparseLUT,
            int *sparseToDenseLUT,
            uint32_t cellAmount
    );

    float initLowestOctreeHierarchy(
            unique_ptr<CudaArray<Chunk>> &octree,
            unique_ptr<CudaArray<uint32_t>> &countingGrid,
            unique_ptr<CudaArray<int>> &denseToSparseLUT,
            unique_ptr<CudaArray<int>> &sparseToDenseLUT,
            uint32_t cellAmount
    );

    __global__ void kernelPropagatePointCounts(
            uint32_t *countingGrid,
            int *denseToSparseLUT,
            uint32_t *sparseIndexCounter,
            uint32_t newCellAmount,
            uint32_t newGridSize,
            uint32_t oldGridSize,
            uint32_t cellOffsetNew,
            uint32_t cellOffsetOld
    );

    float propagatePointCounts(
            unique_ptr<CudaArray<uint32_t>> &countingGrid,
            unique_ptr<CudaArray<int>> &denseToSparseLUT,
            unique_ptr<CudaArray<uint32_t>> &sparseIndexCounter,
            uint32_t newCellAmount,
            uint32_t newGridSize,
            uint32_t oldGridSize,
            uint32_t cellOffsetNew,
            uint32_t cellOffsetOld
    );

    __global__ void kernelMergeHierarchical(
        Chunk *octree,
        uint32_t *countingGrid,
        int *denseToSparseLUT,
        int *sparseToDenseLUT,
        uint32_t *globalChunkCounter,
        uint32_t threshold,
        uint32_t newCellAmount,
        uint32_t newGridSize,
        uint32_t oldGridSize,
        uint32_t cellOffsetNew,
        uint32_t cellOffsetOld
    );

    float mergeHierarchical(
            unique_ptr<CudaArray<Chunk>> &octree,
            unique_ptr<CudaArray<uint32_t>> &countingGrid,
            unique_ptr<CudaArray<int>> &denseToSparseLUT,
            unique_ptr<CudaArray<int>> &sparseToDenseLUT,
            unique_ptr<CudaArray<uint32_t>> &globalChunkCounter,
            uint32_t threshold,
            uint32_t newCellAmount,
            uint32_t newGridSize,
            uint32_t oldGridSize,
            uint32_t cellOffsetNew,
            uint32_t cellOffsetOld
    );


        //_------------- Point Distribution ----------------

    __global__ void kernelDistributePoints (
            Chunk *octree,
            Vector3 *cloud,
            uint32_t *dataLUT,
            int *denseToSparseLUT,
            uint32_t *tmpIndexRegister,
            PointCloudMetadata metadata,
            uint32_t gridSize
    );

    float distributePoints(
        unique_ptr<CudaArray<Chunk>> &octree,
        unique_ptr<CudaArray<Vector3>> &cloud,
        unique_ptr<CudaArray<uint32_t>> &dataLUT,
        unique_ptr<CudaArray<int>> &denseToSparseLUT,
        unique_ptr<CudaArray<uint32_t>> &tmpIndexRegister,
        PointCloudMetadata metadata,
        uint32_t gridSize
    );
}
#endif