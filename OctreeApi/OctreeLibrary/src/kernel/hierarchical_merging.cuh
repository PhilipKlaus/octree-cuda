/**
 * @file hierarchical_merging.cuh
 * @author Philip Klaus
 * @brief Contains code for merging octree nodes in a hierarchical way (bottom-up)
 */

#pragma once

#include "metadata.h"
#include "tools.cuh"
#include "types.cuh"

namespace chunking {

/**
 * Merges octree nodes hierarchically if their point sum is lower than a threshold.
 * The CUDA kernel evaluates the point sum of all 8 child nodes. When the sum is
 * lower than the threshold it 'merges' all 8 child nodes otherwise it marks
 * the parent node and the 8 child nodes as finished.
 *
 * @param octree The octree data structure
 * @param countingGrid Holds the amount of points per cell (dense).
 * @param denseToSparseLUT Holds the dense-to-sparse node mapping.
 * @param sparseToDenseLUT Holds the sparse-to-dense node mapping.
 * @param lutOffset Holds the point-LUT offset.
 * @param threshold The merging threshold.
 * @param cellAmount Cell (node) amount of the current hierarchy.
 * @param gridSize Grid size of the current hierarchy. (e.g. 128)
 * @param lowerGridSize Grid size of one hierarchy level below. (e.g. 256)
 * @param cellOffset The accumulated amount of dense cells for the current hierarchy level.
 * e.g. level=128 -> cellOffset = 512*512*512 + 256*256*256
 * @param cellOffsetLower The accumulated amount of dense cells of one hierarchy level below.
 * e.g. level=128 -> cellOffsetLower = 512*512*512
 */
__global__ void kernelMergeHierarchical (
        Chunk* octree,
        const uint32_t* countingGrid,
        const int* denseToSparseLUT,
        int* sparseToDenseLUT,
        uint32_t* lutOffset,
        uint32_t threshold,
        uint32_t cellAmount,
        uint32_t gridSize,
        uint32_t lowerGridSize,
        uint32_t cellOffset,
        uint32_t cellOffsetLower)
{
    int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

    uint32_t denseVoxelIndex = cellOffset + index; // May be invalid

    if (index >= cellAmount || denseToSparseLUT[denseVoxelIndex] == -1)
    {
        return;
    }

    // Calculate the actual dense coordinates in the octree
    Vector3<uint32_t> coords{};
    tools::mapFromDenseIdxToDenseCoordinates (coords, index, gridSize);

    auto oldXY = lowerGridSize * lowerGridSize;

    // Determine the sparse index in the octree
    int sparseVoxelIndex = denseToSparseLUT[denseVoxelIndex];

    // If the chunk exists, calculate the dense indices of the 8 underlying cells
    uint32_t chunk_0_0_0_index = cellOffsetLower + (coords.z * oldXY * 2) + (coords.y * lowerGridSize * 2) +
                                 (coords.x * 2);                    // int: 0 -> Child 0
    uint32_t chunk_1_0_0_index = chunk_0_0_0_index + 1;             // int: 4 -> child 4
    uint32_t chunk_0_1_0_index = chunk_0_0_0_index + lowerGridSize; // int: 2 -> child 2
    uint32_t chunk_1_1_0_index = chunk_0_1_0_index + 1;             // int: 6 -> child 6
    uint32_t chunk_0_0_1_index = chunk_0_0_0_index + oldXY;         // int: 1 -> child 1
    uint32_t chunk_1_0_1_index = chunk_0_0_1_index + 1;             // int: 5 -> child 5
    uint32_t chunk_0_1_1_index = chunk_0_0_1_index + lowerGridSize; // int: 3 -> child 3
    uint32_t chunk_1_1_1_index = chunk_0_1_1_index + 1;             // int: 7 -> child 7

    // Update the actual (parent) chunk
    Chunk* chunk        = octree + sparseVoxelIndex;
    uint32_t pointCount = countingGrid[denseVoxelIndex];
    bool isFinished     = (pointCount >= threshold);

    // Update the point count
    chunk->pointCount = isFinished ? 0 : pointCount;
    chunk->isParent   = isFinished;

    // Update the isFinished
    chunk->isFinished = isFinished;

    // Assign the sparse indices of the children chunks and calculate the amount of children chunks implicitly
    int sparseChildIndex     = (countingGrid[chunk_0_0_0_index] > 0) ? denseToSparseLUT[chunk_0_0_0_index] : -1;
    chunk->childrenChunks[0] = sparseChildIndex;

    sparseChildIndex         = (countingGrid[chunk_0_0_1_index] > 0) ? denseToSparseLUT[chunk_0_0_1_index] : -1;
    chunk->childrenChunks[1] = sparseChildIndex;

    sparseChildIndex         = (countingGrid[chunk_0_1_0_index] > 0) ? denseToSparseLUT[chunk_0_1_0_index] : -1;
    chunk->childrenChunks[2] = sparseChildIndex;

    sparseChildIndex         = (countingGrid[chunk_0_1_1_index] > 0) ? denseToSparseLUT[chunk_0_1_1_index] : -1;
    chunk->childrenChunks[3] = sparseChildIndex;

    sparseChildIndex         = (countingGrid[chunk_1_0_0_index] > 0) ? denseToSparseLUT[chunk_1_0_0_index] : -1;
    chunk->childrenChunks[4] = sparseChildIndex;

    sparseChildIndex         = (countingGrid[chunk_1_0_1_index] > 0) ? denseToSparseLUT[chunk_1_0_1_index] : -1;
    chunk->childrenChunks[5] = sparseChildIndex;

    sparseChildIndex         = (countingGrid[chunk_1_1_0_index] > 0) ? denseToSparseLUT[chunk_1_1_0_index] : -1;
    chunk->childrenChunks[6] = sparseChildIndex;

    sparseChildIndex         = (countingGrid[chunk_1_1_1_index] > 0) ? denseToSparseLUT[chunk_1_1_1_index] : -1;
    chunk->childrenChunks[7] = sparseChildIndex;

    // Update all children chunks
    auto sum = 0;
    for (auto i = 0; i < 8; ++i)
    {
        if (chunk->childrenChunks[i] != -1)
        {
            // 6.1. Update isFinished in each child
            (octree + chunk->childrenChunks[i])->isFinished = isFinished;

            // 6.3. Assign current sparse chunk index to child as parentChunkIndex
            (octree + chunk->childrenChunks[i])->parentChunkIndex = sparseVoxelIndex;

            sum += (octree + chunk->childrenChunks[i])->pointCount;
        }
    }

    // ##################################################################################


    if (isFinished && sum > 0)
    {
        // Determine the start index inside the dataLUT for all children chunks
        uint32_t dataLUTIndex = atomicAdd (lutOffset, sum);

        for (auto i = 0; i < 8; ++i)
        {
            if (chunk->childrenChunks[i] != -1)
            {
                // 6.2. Update the exact index for the child within the dataLUT
                (octree + chunk->childrenChunks[i])->chunkDataIndex = dataLUTIndex;
                dataLUTIndex += (octree + chunk->childrenChunks[i])->pointCount;
            }
        }
    }

    // Update sparseToDense LUT
    sparseToDenseLUT[sparseVoxelIndex] = denseVoxelIndex;
}
} // namespace chunking