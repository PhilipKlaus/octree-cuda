/**
 * @file point_count_propagation.cuh
 * @author Philip Klaus
 * @brief Contains code for merging point counts in a hierarchical way
 */

#pragma once

#include "metadata.cuh"
#include "tools.cuh"
#include "types.cuh"


namespace chunking {

/**
 * This CUDA kernel propagates point counts from the lowest octree level upwards.
 * Thus the kernel is called for each parent node in each octree level.
 * The kernel sums up the point amounts in all 8 child nodes. If the sum is greater than zero, a new dense-to-sparse
 * entry is created and the filledNodeCounter is increased.
 * The goal is to evaluate the amount of sparse nodes within the octree which is used to allocate the octree.
 *
 * @param countingGrid Holds the amount of points per node.
 * @param denseToSparseLUT Holds the dense-to-sparse node mapping.
 * @param filledNodeCounter Holds the amount of fille nodes (sparse).
 * @param nodeAmount Node amount of the current hierarchy.
 * @param gridSize Grid size of the current hierarchy. (e.g. 128)
 * @param LowerGridSize Grid size of one hierarchy level below. (e.g. 256)
 * @param nodeOffset The accumulated amount of dense nodes for the current hierarchy level.
 * e.g. level=128 -> nodeOffset = 512*512*512 + 256*256*256
 * @param nodeOffsetLower The accumulated amount of dense nodes of one hierarchy level below.
 * e.g. level=128 -> nodeOffsetLower = 512*512*512
 */
__global__ void kernelPropagatePointCounts (
        uint32_t* countingGrid,
        int* denseToSparseLUT,
        uint32_t* filledNodeCounter,
        uint32_t nodeAmount,
        uint32_t gridSize,
        uint32_t lowerGridSize,
        uint32_t nodeOffset,
        uint32_t nodeOffsetLower)
{
    unsigned int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

    if (index >= nodeAmount)
    {
        return;
    }

    // 1. Calculate the actual dense coordinates in the octree
    Vector3<uint32_t> coords{};
    tools::mapFromDenseIdxToDenseCoordinates (coords, index, gridSize);

    auto oldXY = lowerGridSize * lowerGridSize;

    // The new dense index for the actual chunk
    uint32_t denseIndex = nodeOffset + index;

    // Calculate the dense indices of the 8 underlying cells
    uint32_t childNodes[8];
    childNodes[0] = nodeOffsetLower + (coords.z * oldXY * 2) + (coords.y * lowerGridSize * 2) +
                    (coords.x * 2);                // int: 0 -> Child 0
    childNodes[4] = childNodes[0] + 1;             // int: 4 -> child 4
    childNodes[2] = childNodes[0] + lowerGridSize; // int: 2 -> child 2
    childNodes[6] = childNodes[2] + 1;             // int: 6 -> child 6
    childNodes[1] = childNodes[0] + oldXY;         // int: 1 -> child 1
    childNodes[5] = childNodes[1] + 1;             // int: 5 -> child 5
    childNodes[3] = childNodes[1] + lowerGridSize; // int: 3 -> child 3
    childNodes[7] = childNodes[3] + 1;             // int: 7 -> child 7

    // Sum up point counts from all 8 children
    uint32_t sum = 0;
#pragma unroll
    for (uint8_t i = 0; i < 8; ++i)
    {
        sum += *(countingGrid + childNodes[i]);
    }

    if (sum > 0)
    {
        countingGrid[denseIndex] += sum;
        auto sparseIndex             = atomicAdd (filledNodeCounter, 1);
        denseToSparseLUT[denseIndex] = sparseIndex;
    }
}
} // namespace chunking