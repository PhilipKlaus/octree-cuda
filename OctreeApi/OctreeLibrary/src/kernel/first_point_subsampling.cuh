#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <timing.cuh>

#include <cstdint>
#include <global_types.h>
#include <cudaArray.h>
#include <tools.cuh>

namespace subsampling {

    template<typename coordinateType>
    __global__ void kernelFirstPointSubsample(
            uint8_t *cloud,
            SubsampleConfig *subsampleData,
            uint32_t *parentDataLUT,
            uint32_t *countingGrid,
            int *denseToSparseLUT,
            uint32_t *sparseIndexCounter,
            PointCloudMetadata metadata,
            uint32_t gridSideLength,
            uint32_t accumulatedPoints) {


        int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
        if (index >= accumulatedPoints) {
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

        Vector3<coordinateType> *point =
                reinterpret_cast<Vector3<coordinateType> *>(
                        cloud + childDataLUT[childDataLUTStart + index] * metadata.pointDataStride);

        // 1. Calculate the index within the dense grid of the subsample
        auto denseVoxelIndex = tools::calculateGridIndex(point, metadata, gridSideLength);

        // 2. We are only interested in the last point within a node -> Implicitely reset the countingGrid
        auto oldIndex = atomicSub((countingGrid + denseVoxelIndex), 1);

        // 3. If the thread is the first one ->
        //      3.1 store the child lut table index in the parent lut
        //      3.2 'delete' the point within the child lut by invalidating its index entry
        if (oldIndex == 1) {
            parentDataLUT[denseToSparseLUT[denseVoxelIndex]] = childDataLUT[childDataLUTStart + index];
            childDataLUT[childDataLUTStart + index] = INVALID_INDEX;

            // Reset data structures
            denseToSparseLUT[denseVoxelIndex] = 0;
            *sparseIndexCounter = 0;
        }
    }


    template<typename coordinateType>
    float firstPointSubsample(
            unique_ptr<CudaArray<uint8_t>> &cloud,
            unique_ptr<CudaArray<SubsampleConfig>> &subsampleData,
            unique_ptr<CudaArray<uint32_t>> &parentDataLUT,
            unique_ptr<CudaArray<uint32_t>> &countingGrid,
            unique_ptr<CudaArray<int>> &denseToSparseLUT,
            unique_ptr<CudaArray<uint32_t>> &sparseIndexCounter,
            PointCloudMetadata metadata,
            uint32_t gridSideLength,
            uint32_t accumulatedPoints) {

        // Calculate kernel dimensions
        dim3 grid, block;
        tools::create1DKernel(block, grid, accumulatedPoints);

        // Initial point counting
        tools::KernelTimer timer;
        timer.start();
        kernelFirstPointSubsample < coordinateType > << <grid, block >> > (
                cloud->devicePointer(),
                        subsampleData->devicePointer(),
                        parentDataLUT->devicePointer(),
                        countingGrid->devicePointer(),
                        denseToSparseLUT->devicePointer(),
                        sparseIndexCounter->devicePointer(),
                        metadata,
                        gridSideLength,
                        accumulatedPoints);
        timer.stop();
        gpuErrchk(cudaGetLastError());

        spdlog::debug("'kernelFirstPointSubsample' took {:f} [ms]", timer.getMilliseconds());
        return timer.getMilliseconds();
    }
}