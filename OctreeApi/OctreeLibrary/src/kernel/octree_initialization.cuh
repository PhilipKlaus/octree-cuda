#pragma once

#include "octree_metadata.h"
#include "types.cuh"


namespace chunking {

__global__ void kernelOctreeInitialization (
        Chunk* octreeSparse,
        uint32_t* densePointCount,
        int* denseToSparseLUT,
        int* sparseToDenseLUT,
        uint32_t cellAmount)
{
    int denseVoxelIndex = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

    if (denseVoxelIndex >= cellAmount)
    {
        return;
    }

    int sparseVoxelIndex = denseToSparseLUT[denseVoxelIndex];

    if (sparseVoxelIndex == -1)
    {
        return;
    }

    // Update sparseToDense LUT
    sparseToDenseLUT[sparseVoxelIndex] = denseVoxelIndex;

    Chunk* chunk      = octreeSparse + sparseVoxelIndex;
    chunk->pointCount = densePointCount[denseVoxelIndex];

    chunk->childrenChunks[0] = -1;
    chunk->childrenChunks[1] = -1;
    chunk->childrenChunks[2] = -1;
    chunk->childrenChunks[3] = -1;
    chunk->childrenChunks[4] = -1;
    chunk->childrenChunks[5] = -1;
    chunk->childrenChunks[6] = -1;
    chunk->childrenChunks[7] = -1;

    assert (chunk->pointCount != 0);
}

} // namespace chunking
