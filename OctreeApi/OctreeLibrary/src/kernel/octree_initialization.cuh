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
        Node* octreeSparse,
        const uint32_t* countingGrid,
        const int* denseToSparseLUT,
        int* sparseToDenseLUT,
        uint32_t nodeAmount)
{
    unsigned int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

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

    Node* node             = octreeSparse + sparseVoxelIndex;
    node->pointCount        = countingGrid[index];

#pragma unroll
    for(auto i = 0; i < 8; ++i) {
        node->childNodes[i] = -1;
    }
}

} // namespace chunking
