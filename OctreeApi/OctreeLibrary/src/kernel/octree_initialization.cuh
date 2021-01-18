#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <timing.cuh>

#include <memory>
#include <global_types.h>
#include <types.cuh>
#include <cudaArray.h>
#include <tools.cuh>


namespace chunking {

    __global__ void kernelOctreeInitialization(
            Chunk *octreeSparse,
            uint32_t *densePointCount,
            int *denseToSparseLUT,
            int *sparseToDenseLUT,
            uint32_t cellAmount
    ) {

        int denseVoxelIndex = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

        if(denseVoxelIndex >= cellAmount) {
            return;
        }

        int sparseVoxelIndex = denseToSparseLUT[denseVoxelIndex];

        if(sparseVoxelIndex == -1) {
            return;
        }

        // Update sparseToDense LUT
        sparseToDenseLUT[sparseVoxelIndex] = denseVoxelIndex;

        Chunk *chunk = octreeSparse + sparseVoxelIndex;
        chunk->pointCount = densePointCount[denseVoxelIndex];

        chunk->childrenChunks[0] = -1;
        chunk->childrenChunks[1] = -1;
        chunk->childrenChunks[2] = -1;
        chunk->childrenChunks[3] = -1;
        chunk->childrenChunks[4] = -1;
        chunk->childrenChunks[5] = -1;
        chunk->childrenChunks[6] = -1;
        chunk->childrenChunks[7] = -1;

        assert(chunk->pointCount != 0);
    }

    float initOctree(
            GpuOctree &octree,
            GpuArrayU32 &countingGrid,
            GpuArrayI32 &denseToSparseLUT,
            GpuArrayI32 &sparseToDenseLUT,
            uint32_t lowestGridSize
    ) {

        dim3 grid, block;
        tools::create1DKernel(block, grid, lowestGridSize);

        tools::KernelTimer timer;
        timer.start();
        chunking::kernelOctreeInitialization << < grid, block >> > (
                octree->devicePointer(),
                        countingGrid->devicePointer(),
                        denseToSparseLUT->devicePointer(),
                        sparseToDenseLUT->devicePointer(),
                        lowestGridSize);
        timer.stop();
        gpuErrchk(cudaGetLastError());
        return timer.getMilliseconds();
    }

}
