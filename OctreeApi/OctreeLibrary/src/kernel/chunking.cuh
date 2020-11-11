#include <cuda_runtime.h>
#include <types.h>
#include <cudaArray.h>

namespace chunking {

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
}