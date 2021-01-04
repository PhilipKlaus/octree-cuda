#include <cuda_runtime.h>
#include <cuda.h>
#include <timing.cuh>

#include <cstdint>
#include <types.h>
#include <cudaArray.h>
#include <tools.cuh>

#include <curand_kernel.h>

namespace subsampling {

    template <typename coordinateType>
    __global__ void kernelEvaluateSubsamples(
            uint8_t *cloud,
            uint32_t *cloudDataLUT,
            uint32_t dataLUTStartIndex,
            uint32_t *densePointCount,
            int *denseToSparseLUT,
            uint32_t *sparseIndexCounter,
            PointCloudMetadata metadata,
            uint32_t gridSideLength) {

        int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
        if(index >= metadata.pointAmount) {
            return;
        }

        CoordinateVector<coordinateType> *point =
                reinterpret_cast<CoordinateVector<coordinateType>*>(
                        cloud + cloudDataLUT[dataLUTStartIndex + index] * metadata.pointDataStride);

        // 1. Calculate the index within the dense grid of the evaluateSubsamples
        auto denseVoxelIndex = tools::calculateGridIndex(point, metadata, gridSideLength);

        // 2. We are only interested in the first point within a cell
        auto oldIndex = atomicAdd((densePointCount + denseVoxelIndex), 1);

        // 3. If the thread is the first one -> increase map from the dense grid to the sparse grid
        if(oldIndex == 0) {
            auto sparseVoxelIndex = atomicAdd(sparseIndexCounter, 1);
            denseToSparseLUT[denseVoxelIndex] = sparseVoxelIndex;
        }
    }


    template <typename coordinateType>
    float evaluateSubsamples(
            unique_ptr<CudaArray<uint8_t>> &cloud,
            unique_ptr<CudaArray<uint32_t>> &cloudDataLUT,
            uint32_t dataLUTStartIndex,
            unique_ptr<CudaArray<uint32_t>> &countingGrid,
            unique_ptr<CudaArray<int>> &denseToSparseLUT,
            unique_ptr<CudaArray<uint32_t>> &sparseIndexCounter,
            PointCloudMetadata metadata,
            uint32_t gridSideLength) {

        // Calculate kernel dimensions
        dim3 grid, block;
        tools::create1DKernel(block, grid, metadata.pointAmount);

        // Initial point counting
        tools::KernelTimer timer;
        timer.start();


        kernelEvaluateSubsamples<coordinateType> << < grid, block >> > (
                cloud->devicePointer(),
                        cloudDataLUT->devicePointer(),
                        dataLUTStartIndex,
                        countingGrid->devicePointer(),
                        denseToSparseLUT->devicePointer(),
                        sparseIndexCounter->devicePointer(),
                        metadata,
                        gridSideLength);
        timer.stop();
        gpuErrchk(cudaGetLastError());

        spdlog::debug("'kernelEvaluateSubsamples' took {:f} [ms]", timer.getMilliseconds());
        return timer.getMilliseconds();
    }

}
