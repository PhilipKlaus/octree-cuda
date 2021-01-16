#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <timing.cuh>

#include <cstdint>
#include <types.h>
#include <cudaArray.h>
#include <tools.cuh>

#include <curand_kernel.h>
#include "../../include/types.h"

namespace subsampling {

    template <typename coordinateType, typename colorType>
    __global__ void kernelPerformAveraging(
            uint8_t *cloud,
            SubsampleConfig *subsampleData,
            Averaging *parentAveragingData,
            int *denseToSparseLUT,
            PointCloudMetadata metadata,
            uint32_t gridSideLength,
            uint32_t accumulatedPoints) {


        int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
        if(index >= accumulatedPoints) {
            return;
        }

        // Determine child index and pick appropriate LUT data
        uint32_t *childDataLUT = nullptr;
        Averaging *childAveraging = nullptr;
        uint32_t childDataLUTStart = 0;

        for(int i = 0; i < 8; ++i) {
            if(index < subsampleData[i].pointOffsetUpper) {
                childDataLUT = subsampleData[i].lutAdress;
                childAveraging = subsampleData[i].averagingAdress;
                childDataLUTStart = subsampleData[i].lutStartIndex;
                index -= subsampleData[i].pointOffsetLower;
                break;
            }
        }

        // Get the coordinates from the point within the point cloud
        CoordinateVector<coordinateType> *point =
                reinterpret_cast<CoordinateVector<coordinateType>*>(
                        cloud + childDataLUT[childDataLUTStart + index] * metadata.pointDataStride);

        // Get the color from the point within the point cloud
        CoordinateVector<colorType> *color =
                reinterpret_cast<CoordinateVector<colorType>*>(
                        cloud +
                        childDataLUT[childDataLUTStart + index] * metadata.pointDataStride
                        + sizeof(coordinateType) * 3);

        // 1. Calculate the index within the dense grid of the evaluateSubsamples
        auto denseVoxelIndex = tools::calculateGridIndex(point, metadata, gridSideLength);

        int sparseIndex = denseToSparseLUT[denseVoxelIndex];

        bool hasAveragingData = (childAveraging != nullptr);
        atomicAdd(&(parentAveragingData[sparseIndex].pointCount), hasAveragingData ?  childAveraging[index].pointCount : 1);

        atomicAdd(&(parentAveragingData[sparseIndex].r), hasAveragingData ?  childAveraging[index].r : color->x);
        atomicAdd(&(parentAveragingData[sparseIndex].g), hasAveragingData ?  childAveraging[index].g : color->y);
        atomicAdd(&(parentAveragingData[sparseIndex].b), hasAveragingData ?  childAveraging[index].b : color->z);
    }


    // Move point indices from old (child LUT) to new (parent LUT)
    template <typename coordinateType>
    __global__ void kernelRandomPointSubsample(
            uint8_t *cloud,
            SubsampleConfig *subsampleData,
            uint32_t *parentDataLUT,
            Averaging *averagingData,
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

        // Get the point within the point cloud
        CoordinateVector<coordinateType> *point =
                reinterpret_cast<CoordinateVector<coordinateType>*>(
                        cloud + childDataLUT[childDataLUTStart + index] * metadata.pointDataStride);

        // 1. Calculate the index within the dense grid of the evaluateSubsamples
        auto denseVoxelIndex = tools::calculateGridIndex(point, metadata, gridSideLength);

        int sparseIndex = denseToSparseLUT[denseVoxelIndex];

        // 2. We are only interested in the last point within a node -> Implicitly reset the countingGrid
        auto oldIndex = atomicSub((countingGrid + denseVoxelIndex), 1);

        if(sparseIndex == -1 || oldIndex != randomIndices[sparseIndex]) {
            return;
        }

        // Move subsampled point to parent
        parentDataLUT[sparseIndex] = childDataLUT[childDataLUTStart + index];
        //childDataLUT[childDataLUTStart + index] = INVALID_INDEX; // Replacement strategy

        // Reset all subsampling data data
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
            Averaging *averagingData,
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

        // Generate random value for point picking
        randomIndices[sparseIndex] = static_cast<uint32_t>(ceil(curand_uniform(&states[threadIdx.x]) * countingGrid[index]));

        // Reset Averaging data
        averagingData[sparseIndex].r = 0.f;
        averagingData[sparseIndex].g = 0.f;
        averagingData[sparseIndex].b = 0.f;
        averagingData[sparseIndex].pointCount = 0;
    }

    template <typename coordinateType, typename colorType>
    float performAveraging(
            unique_ptr<CudaArray<uint8_t>> &cloud,
            unique_ptr<CudaArray<SubsampleConfig>> &subsampleData,
            unique_ptr<CudaArray<Averaging>>& parentAveragingData,
            unique_ptr<CudaArray<int>> &denseToSparseLUT,
            PointCloudMetadata metadata,
            uint32_t gridSideLength,
            uint32_t accumulatedPoints) {

        // Calculate kernel dimensions
        dim3 grid, block;
        tools::create1DKernel(block, grid, accumulatedPoints);

        // Initial point counting
        tools::KernelTimer timer;
        timer.start();
        kernelPerformAveraging < coordinateType, colorType > << < grid, block >> > (
                        cloud->devicePointer(),
                        subsampleData->devicePointer(),
                        parentAveragingData->devicePointer(),
                        denseToSparseLUT->devicePointer(),
                        metadata,
                        gridSideLength,
                        accumulatedPoints);
        timer.stop();
        gpuErrchk(cudaGetLastError());

        spdlog::debug("'kernelPerformAveraging' took {:f} [ms]", timer.getMilliseconds());
        return timer.getMilliseconds();
    }
    template <typename coordinateType>
    float randomPointSubsample(
            unique_ptr<CudaArray<uint8_t>> &cloud,
            unique_ptr<CudaArray<SubsampleConfig>> &subsampleData,
            unique_ptr<CudaArray<uint32_t>> &parentDataLUT,
            unique_ptr<CudaArray<Averaging>> &averagingData,
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
                        averagingData->devicePointer(),
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
            const unique_ptr<CudaArray<Averaging>> &averagingData,
            const unique_ptr<CudaArray<uint32_t>> &countingGrid,
            uint32_t gridNodes) {

        // Calculate kernel dimensions
        dim3 grid, block;
        tools::create1DKernel(block, grid, gridNodes);

        // Initial point counting
        tools::KernelTimer timer;
        timer.start();
        kernelGenerateRandoms << < grid, block >> > (
                states->devicePointer(),
                randomIndices->devicePointer(),
                denseToSparseLUT->devicePointer(),
                averagingData->devicePointer(),
                countingGrid->devicePointer(),
                gridNodes);
        timer.stop();
        gpuErrchk(cudaGetLastError());

        spdlog::debug("'kernelGenerateRandoms' took {:f} [ms]", timer.getMilliseconds());
        return timer.getMilliseconds();
    }

    //float resetAveragingData()
}