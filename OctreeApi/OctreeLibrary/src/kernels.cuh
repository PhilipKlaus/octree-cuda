#include <cuda_runtime.h>
#include "../include/types.h"
#include "../include/cudaArray.h"

#ifndef OCTREECUDA_KERNELS_CUH
#define OCTREECUDA_KERNELS_CUH

namespace kernel {
    /**
     * This kernel maps 3D points to a 3-dimensional grid (gridSideLength x gridSideLength x gridSideLength).
     *
     * @param[in] cloud
     * @param[out] densePointCount Holds the amount of points which fall into each single cell.
     * @param[out] denseToSparseLUT A Mapping from the dense grid to a continuous sparse grid.
     * @param[out] sparseIndexCounter Denotes the amount of entries in the denseToSparseLUT.
     * @param[in] metadata The metadata for the point count.
     * @param[in] gridSize The length of one grid side in voxels.
     * @return void
     */
    __global__ void kernelMapCloudToGrid(
            Vector3 *cloud,
            uint32_t *densePointCount,
            int *denseToSparseLUT,
            uint32_t *sparseIndexCounter,
            PointCloudMetadata metadata,
            uint32_t gridSideLength
    );

    /**
     * This kernel maps 3D points to a 3-dimensional grid (gridSideLength x gridSideLength x gridSideLength).
     * This kernel accesses the 3D points by using the provided data LUT.
     *
     * @param[in] cloud
     * @param[in] cloudDataLUT The LUT for accessing the 3D points within the cloud.
     * @param[in] dataLUTStartIndex The start index within the data LUT.
     * @param[out] densePointCount Holds the amount of points which fall into each single cell.
     * @param[out] denseToSparseLUT A Mapping from the dense grid to a continuous sparse grid.
     * @param[out] sparseIndexCounter Denotes the amount of entries in the denseToSparseLUT.
     * @param[in] metadata The metadata for the point count.
     * @param[in] gridSize The length of one grid side in voxels.
     * @return void
     */
    __global__ void kernelMapCloudToGrid_LUT(
            Vector3 *cloud,
            uint32_t *cloudDataLUT,
            uint32_t dataLUTStartIndex,
            uint32_t *densePointCount,
            int *denseToSparseLUT,
            uint32_t *sparseIndexCounter,
            PointCloudMetadata metadata,
            uint32_t gridSideLength
    );

    /*__global__ void kernelDistributeSubsamlePoints(
            Vector3 *cloud,
            uint32_t *cloudDataLUT,
            uint32_t dataLUTStartIndex,
            uint32_t *countingGrid,
            int *denseToSparseLUT,
            uint32_t *sparseIndexCounter,
            PointCloudMetadata metadata,
            uint32_t gridSideLength
    );*/

}

namespace kernelExecution {

    float executeKernelMapCloudToGrid(
            unique_ptr<CudaArray<Vector3>> &cloud,
            unique_ptr<CudaArray<uint32_t>> &densePointCount,
            unique_ptr<CudaArray<int>> &denseToSparseLUT,
            unique_ptr<CudaArray<uint32_t>> &sparseIndexCounter,
            PointCloudMetadata metadata,
            uint32_t gridSideLength
            );

    float executeKernelMapCloudToGrid_LUT(
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