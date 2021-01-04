#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <timing.cuh>

#include <cstdint>
#include <types.h>
#include <cudaArray.h>
#include <tools.cuh>

namespace chunking {

    template <typename coordinateType>
    __global__ void kernelInitialPointCounting(
            uint8_t *cloud,
            uint32_t *densePointCount,
            int *denseToSparseLUT,
            uint32_t *sparseIndexCounter,
            PointCloudMetadata metadata,
            uint32_t gridSideLength
    ) {

        int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
        if(index >= metadata.pointAmount) {
            return;
        }

        CoordinateVector<coordinateType> *point =
                reinterpret_cast<CoordinateVector<coordinateType>*>(cloud + index * metadata.pointDataStride);

        // 1. Calculate the index within the dense grid
        auto denseVoxelIndex = tools::calculateGridIndex<coordinateType>(point, metadata, gridSideLength);

        // 2. Accumulate the counter within the dense cell
        auto oldIndex = atomicAdd((densePointCount + denseVoxelIndex), 1);

        // 3. If the thread is the first one accumulating the counter within the cell -> update the denseToSparseLUT
        if(oldIndex == 0) {
            auto sparseVoxelIndex = atomicAdd(sparseIndexCounter, 1);
            denseToSparseLUT[denseVoxelIndex] = sparseVoxelIndex;
        }
    }


    template <typename coordinateType>
    float initialPointCounting(
            unique_ptr<CudaArray<uint8_t>> &cloud,
            unique_ptr<CudaArray<uint32_t>> &densePointCount,
            unique_ptr<CudaArray<int>> &denseToSparseLUT,
            unique_ptr<CudaArray<uint32_t>> &sparseIndexCounter,
            PointCloudMetadata metadata,
            uint32_t gridSideLength
    ) {

        // Calculate kernel dimensions
        dim3 grid, block;
        tools::create1DKernel(block, grid, metadata.pointAmount);

        // Initial point counting
        tools::KernelTimer timer;
        timer.start();
        chunking::kernelInitialPointCounting<coordinateType> <<<  grid, block >>> (
                cloud->devicePointer(),
                        densePointCount->devicePointer(),
                        denseToSparseLUT->devicePointer(),
                        sparseIndexCounter->devicePointer(),
                        metadata,
                        gridSideLength);
        timer.stop();
        gpuErrchk(cudaGetLastError());
        return timer.getMilliseconds();
    }
}