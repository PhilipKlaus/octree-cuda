#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <timing.cuh>

#include <cstdint>
#include <types.h>
#include <cudaArray.h>
#include <tools.cuh>

#include <curand_kernel.h>

namespace subsampling {

    // Move point indices from old (child LUT) to new (parent LUT)
    template <typename coordinateType>
    __global__ void kernelRandomPointSubsample(
            uint8_t *cloud,
            SubsampleConfig *subsampleData,
            uint32_t *parentDataLUT,
            uint32_t *countingGrid,
            int *denseToSparseLUT,
            uint32_t *sparseIndexCounter,
            PointCloudMetadata metadata,
            uint32_t gridSideLength,
            uint32_t *randomIndices,
            uint32_t accumulatedPoints) {


        int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
        if(index >= accumulatedPoints) {
            return;
        }

        // Determine child index and pick appropriate LUT data
        uint32_t *childDataLUT = nullptr;
        uint32_t childDataLUTStart = 0;

        for(int i = 0; i < 8; ++i) {
            if(index < subsampleData[i].pointOffsetUpper) {
                childDataLUT = subsampleData[i].lutAdress;
                childDataLUTStart = subsampleData[i].lutStartIndex;
                index -= subsampleData[i].pointOffsetLower;
                break;
            }
        }

        CoordinateVector<coordinateType> *point =
                reinterpret_cast<CoordinateVector<coordinateType>*>(
                        cloud + childDataLUT[childDataLUTStart + index] * metadata.pointDataStride);

        // 1. Calculate the index within the dense grid of the evaluateSubsamples
        auto denseVoxelIndex = tools::calculateGridIndex(point, metadata, gridSideLength);

        // 2. We are only interested in the last point within a node -> Implicitly reset the countingGrid
        auto oldIndex = atomicSub((countingGrid + denseVoxelIndex), 1);

        // 3. If the thread is the first one ->
        //      3.1 store the child lut table index in the parent lut
        //      3.2 'delete' the point within the child lut by invalidating its index entry
        int sparseIndex = denseToSparseLUT[denseVoxelIndex];

        if(sparseIndex == -1 || oldIndex != randomIndices[sparseIndex]) {
            return;
        }

        parentDataLUT[sparseIndex] = childDataLUT[childDataLUTStart + index];
        childDataLUT[childDataLUTStart + index] = INVALID_INDEX;
        denseToSparseLUT[denseVoxelIndex] = -1;
        *sparseIndexCounter = 0;
    }

    // http://ianfinlayson.net/class/cpsc425/notes/cuda-random
    __global__ void kernelInitRandoms(
            unsigned int seed,
            curandState_t* states,
            uint32_t nodeAmount) {

        int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

        if(index >= nodeAmount) {
            return;
        }

        curand_init(seed, index, 0, &states[index]);
    }

    __global__ void kernelGenerateRandoms(
            curandState_t* states,
            uint32_t *randomIndices,
            const int *denseToSparseLUT,
            const uint32_t *countingGrid,
            uint32_t gridNodes) {

        int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

        if(index >= gridNodes) {
            return;
        }

        int sparseIndex = denseToSparseLUT[index];

        if(sparseIndex == -1) {
            return;
        }

        randomIndices[sparseIndex] = static_cast<uint32_t>(ceil(curand_uniform(&states[threadIdx.x]) * countingGrid[index]));
    }

    template <typename coordinateType>
    float randomPointSubsample(
            unique_ptr<CudaArray<uint8_t>> &cloud,
            unique_ptr<CudaArray<SubsampleConfig>> &subsampleData,
            unique_ptr<CudaArray<uint32_t>> &parentDataLUT,
            unique_ptr<CudaArray<uint32_t>> &countingGrid,
            unique_ptr<CudaArray<int>> &denseToSparseLUT,
            unique_ptr<CudaArray<uint32_t>> &sparseIndexCounter,
            PointCloudMetadata metadata,
            uint32_t gridSideLength,
            unique_ptr<CudaArray<uint32_t>> &randomIndices,
            uint32_t accumulatedPoints) {

        // Calculate kernel dimensions
        dim3 grid, block;
        tools::create1DKernel(block, grid, accumulatedPoints);

        // Initial point counting
        tools::KernelTimer timer;
        timer.start();
        kernelRandomPointSubsample < coordinateType > << < grid, block >> > (
                cloud->devicePointer(),
                        subsampleData->devicePointer(),
                        parentDataLUT->devicePointer(),
                        countingGrid->devicePointer(),
                        denseToSparseLUT->devicePointer(),
                        sparseIndexCounter->devicePointer(),
                        metadata,
                        gridSideLength,
                        randomIndices->devicePointer(),
                        accumulatedPoints);
        timer.stop();
        gpuErrchk(cudaGetLastError());

        spdlog::debug("'kernelRandomPointSubsample' took {:f} [ms]", timer.getMilliseconds());
        return timer.getMilliseconds();
    }


    float initRandoms(
            unsigned int seed,
            unique_ptr<CudaArray<curandState_t>> &states,
            uint32_t nodeAmount) {

        // Calculate kernel dimensions
        dim3 grid, block;
        tools::create1DKernel(block, grid, nodeAmount);

        // Initial point counting
        tools::KernelTimer timer;
        timer.start();
        kernelInitRandoms << < grid, block >> > (seed, states->devicePointer(), nodeAmount);
        timer.stop();
        gpuErrchk(cudaGetLastError());

        spdlog::debug("'kernelInitRandoms' took {:f} [ms]", timer.getMilliseconds());
        return timer.getMilliseconds();
    }

    float generateRandoms(
            const unique_ptr<CudaArray<curandState_t>> &states,
            unique_ptr<CudaArray<uint32_t>> &randomIndices,
            const unique_ptr<CudaArray<int>> &denseToSparseLUT,
            const unique_ptr<CudaArray<uint32_t>> &countingGrid,
            uint32_t gridNodes) {

        // Calculate kernel dimensions
        dim3 grid, block;
        tools::create1DKernel(block, grid, gridNodes);

        // Initial point counting
        tools::KernelTimer timer;
        timer.start();
        kernelGenerateRandoms << < grid, block >> > (states->devicePointer(), randomIndices->devicePointer(), denseToSparseLUT->devicePointer(), countingGrid->devicePointer(), gridNodes);
        timer.stop();
        gpuErrchk(cudaGetLastError());

        spdlog::debug("'kernelGenerateRandoms' took {:f} [ms]", timer.getMilliseconds());
        return timer.getMilliseconds();
    }

}