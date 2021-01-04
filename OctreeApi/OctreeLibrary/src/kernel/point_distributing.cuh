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
    __global__ void kernelDistributePoints (
            Chunk *octree,
            uint8_t *cloud,
            uint32_t *dataLUT,
            int *denseToSparseLUT,
            uint32_t *tmpIndexRegister,
            PointCloudMetadata metadata,
            uint32_t gridSize
    ) {

        int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
        if(index >= metadata.pointAmount) {
            return;
        }

        CoordinateVector<coordinateType> *point =
                reinterpret_cast<CoordinateVector<coordinateType>*>(cloud + index * metadata.pointDataStride);

        auto denseVoxelIndex = tools::calculateGridIndex(point, metadata, gridSize);
        auto sparseVoxelIndex = denseToSparseLUT[denseVoxelIndex];

        bool isFinished = octree[sparseVoxelIndex].isFinished;

        while(!isFinished) {
            sparseVoxelIndex = octree[sparseVoxelIndex].parentChunkIndex;
            isFinished = octree[sparseVoxelIndex].isFinished;
        }

        uint32_t dataIndexWithinChunk = atomicAdd(tmpIndexRegister + sparseVoxelIndex, 1);
        dataLUT[octree[sparseVoxelIndex].chunkDataIndex + dataIndexWithinChunk] = index;
    }

    template <typename coordinateType>
    float distributePoints(
            unique_ptr<CudaArray<Chunk>> &octree,
            unique_ptr<CudaArray<uint8_t>> &cloud,
            unique_ptr<CudaArray<uint32_t>> &dataLUT,
            unique_ptr<CudaArray<int>> &denseToSparseLUT,
            unique_ptr<CudaArray<uint32_t>> &tmpIndexRegister,
            PointCloudMetadata metadata,
            uint32_t gridSize
    ) {

        // Calculate kernel dimensions
        dim3 grid, block;
        tools::create1DKernel(block, grid, metadata.pointAmount);

        // Call distribution kernel
        tools::KernelTimer timer;
        timer.start();
        chunking::kernelDistributePoints<coordinateType> <<<  grid, block >>> (
                octree->devicePointer(),
                        cloud->devicePointer(),
                        dataLUT->devicePointer(),
                        denseToSparseLUT->devicePointer(),
                        tmpIndexRegister->devicePointer(),
                        metadata,
                        gridSize);
        timer.stop();
        gpuErrchk(cudaGetLastError());
        return timer.getMilliseconds();
    }

}