#pragma once

#include "octree_metadata.h"
#include "types.cuh"


namespace chunking {

__global__ void kernelOctreeInitialization (
        Chunk* octreeSparse,
        const uint32_t* densePointCount,
        const int* denseToSparseLUT,
        int* sparseToDenseLUT,
        uint32_t nodeAmount)
{
    int indexDense = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

    if (indexDense >= nodeAmount)
    {
        return;
    }

    int sparseVoxelIndex = denseToSparseLUT[indexDense];

    if (sparseVoxelIndex == -1)
    {
        return;
    }

    sparseToDenseLUT[sparseVoxelIndex] = indexDense;

    Chunk* chunk      = octreeSparse + sparseVoxelIndex;
    chunk->pointCount = densePointCount[indexDense];

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
