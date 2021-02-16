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
        uint32_t LowerGridSize,
        uint32_t cellOffset,
        uint32_t cellOffsetLower)
{
    int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

    if (index >= cellAmount)
    {
        return;
    }

    // 1. Calculate the actual dense coordinates in the octree
    Vector3<uint32_t> coords{};
    tools::mapFromDenseIdxToDenseCoordinates (coords, index, gridSize);

    auto oldXY = LowerGridSize * LowerGridSize;

    // The new dense index for the actual chunk
    uint32_t denseIndex = cellOffset + index;

    // Calculate the dense indices of the 8 underlying cells
    uint32_t chunk_0_0_0_index = cellOffsetLower + (coords.z * oldXY * 2) + (coords.y * LowerGridSize * 2) +
                                 (coords.x * 2);                    // int: 0 -> Child 0
    uint32_t chunk_1_0_0_index = chunk_0_0_0_index + 1;             // int: 4 -> child 4
    uint32_t chunk_0_0_1_index = chunk_0_0_0_index + LowerGridSize; // int: 1 -> child 1
    uint32_t chunk_1_0_1_index = chunk_0_0_1_index + 1;             // int: 5 -> child 5
    uint32_t chunk_0_1_0_index = chunk_0_0_0_index + oldXY;         // int: 2 -> child 2
    uint32_t chunk_1_1_0_index = chunk_0_1_0_index + 1;             // int: 6 -> child 6
    uint32_t chunk_0_1_1_index = chunk_0_1_0_index + LowerGridSize; // int: 3 -> child 3
    uint32_t chunk_1_1_1_index = chunk_0_1_1_index + 1;             // int: 7 -> child 7

    // Create pointers to the 8 underlying cells
    uint32_t* chunk_0_0_0 = countingGrid + chunk_0_0_0_index;
    uint32_t* chunk_0_0_1 = countingGrid + chunk_0_0_1_index;
    uint32_t* chunk_0_1_0 = countingGrid + chunk_0_1_0_index;
    uint32_t* chunk_0_1_1 = countingGrid + chunk_0_1_1_index;
    uint32_t* chunk_1_0_0 = countingGrid + chunk_1_0_0_index;
    uint32_t* chunk_1_0_1 = countingGrid + chunk_1_0_1_index;
    uint32_t* chunk_1_1_0 = countingGrid + chunk_1_1_0_index;
    uint32_t* chunk_1_1_1 = countingGrid + chunk_1_1_1_index;

    // Buffer the point counts within each cell
    uint32_t chunk_0_0_0_count = *chunk_0_0_0;
    uint32_t chunk_0_0_1_count = *chunk_0_0_1;
    uint32_t chunk_0_1_0_count = *chunk_0_1_0;
    uint32_t chunk_0_1_1_count = *chunk_0_1_1;
    uint32_t chunk_1_0_0_count = *chunk_1_0_0;
    uint32_t chunk_1_0_1_count = *chunk_1_0_1;
    uint32_t chunk_1_1_0_count = *chunk_1_1_0;
    uint32_t chunk_1_1_1_count = *chunk_1_1_1;

    // Summarize all children counts
    auto sum = chunk_0_0_0_count + chunk_0_0_1_count + chunk_0_1_0_count + chunk_0_1_1_count + chunk_1_0_0_count +
               chunk_1_0_1_count + chunk_1_1_0_count + chunk_1_1_1_count;

    if (sum > 0)
    {
        countingGrid[denseIndex] += sum;
        auto sparseIndex             = atomicAdd (filledNodeCounter, 1);
        denseToSparseLUT[denseIndex] = sparseIndex;
    }
}
} // namespace chunking