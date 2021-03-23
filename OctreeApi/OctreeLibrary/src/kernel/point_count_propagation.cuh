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
 * This CUDA kernel is executed for each potential parent node (cell).
 * The kernel evaluates the point counts of all its children nodes (cells).
 * If the sum is higher than zero, the sparse index of the parent node (cell)
 * is added to the dense-to-sparse LUT for further processing.
 * After the propagation the actual sparse Node amount is known and the octree
 * datastructure can be allocated.
 *
 * @param countingGrid Holds the amount of points per node (cell).
 * @param denseToSparseLUT Holds the dense-to-sparse node mapping.
 * @param filledNodeCounter Holds the amount of filled (non-empty) cells (sparse).
 * @param cellAmount Cell (node) amount of the current hierarchy.
 * @param gridSize Grid size of the current hierarchy. (e.g. 128)
 * @param LowerGridSize Grid size of one hierarchy level below. (e.g. 256)
 * @param cellOffset The accumulated amount of dense cells for the current hierarchy level.
 * e.g. level=128 -> cellOffset = 512*512*512 + 256*256*256
 * @param cellOffsetLower The accumulated amount of dense cells of one hierarchy level below.
 * e.g. level=128 -> cellOffsetLower = 512*512*512
 */
__global__ void kernelPropagatePointCounts (
        uint32_t* countingGrid,
        int* denseToSparseLUT,
        uint32_t* filledNodeCounter,
        uint32_t cellAmount,
        uint32_t gridSize,
        uint32_t lowerGridSize,
        uint32_t cellOffset,
        uint32_t cellOffsetLower)
{
    unsigned int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

    if (index >= cellAmount)
    {
        return;
    }

    // 1. Calculate the actual dense coordinates in the octree
    Vector3<uint32_t> coords{};
    tools::mapFromDenseIdxToDenseCoordinates (coords, index, gridSize);

    auto oldXY = lowerGridSize * lowerGridSize;

    // The new dense index for the actual chunk
    uint32_t denseIndex = cellOffset + index;

    // Calculate the dense indices of the 8 underlying cells
    uint32_t childNodes[8];
    childNodes[0] = cellOffsetLower + (coords.z * oldXY * 2) + (coords.y * lowerGridSize * 2) +
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