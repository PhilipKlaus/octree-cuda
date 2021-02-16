/**
 * @file octree_initialization.cuh
 * @author Philip Klaus
 * @brief Contains code for initializing the base level (leafs) of the octree.
 */

#pragma once

#include "metadata.cuh"
#include "types.cuh"


namespace chunking {

/**
 * Initializes the leaf nodes of the octree and adds sparse-to-dense mappings.
 *
 * @param octreeSparse The octree datastructure.
 * @param countingGrid Holds the amount of points per cell (dense).
 * @param denseToSparseLUT Holds the dense-to-sparse node mapping.
 * @param sparseToDenseLUT Holds the sparse-to-dense node mapping.
 * @param nodeAmount The maximum amount of nodees (cells).
 */
__global__ void kernelInitLeafNodes (
        Chunk* octreeSparse,
        const uint32_t* countingGrid,
        const int* denseToSparseLUT,
        int* sparseToDenseLUT,
        uint32_t nodeAmount)
{
    int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

    if (index >= nodeAmount)
    {
        return;
    }

    int sparseVoxelIndex = denseToSparseLUT[index];

    if (sparseVoxelIndex == -1)
    {
        return;
    }

    sparseToDenseLUT[sparseVoxelIndex] = index;

    Chunk* chunk             = octreeSparse + sparseVoxelIndex;
    chunk->pointCount        = countingGrid[index];
    chunk->childrenChunks[0] = -1;
    chunk->childrenChunks[1] = -1;
    chunk->childrenChunks[2] = -1;
    chunk->childrenChunks[3] = -1;
    chunk->childrenChunks[4] = -1;
    chunk->childrenChunks[5] = -1;
    chunk->childrenChunks[6] = -1;
    chunk->childrenChunks[7] = -1;
}

} // namespace chunking
