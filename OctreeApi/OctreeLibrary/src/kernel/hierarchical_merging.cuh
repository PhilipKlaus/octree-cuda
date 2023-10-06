/**
 * @file hierarchical_merging.cuh
 * @author Philip Klaus
 * @brief Contains code for merging octree nodes in a hierarchical way (bottom-up)
 */

#pragma once

#include "metadata.cuh"
#include "tools.cuh"
#include "types.cuh"

namespace chunking {

/**
 * Merges octree nodes hierarchically if their point sum is lower than a threshold.
 * The CUDA kernel evaluates the point sum of all 8 child nodes. When the sum is
 * lower than the threshold it 'merges' all 8 child nodes, otherwise it marks
 * the parent node and the 8 child nodes as finished.
 *
 * @param octree The octree data structure
 * @param countingGrid The amount of points per node.
 * @param denseToSparseLUT Holds the dense-to-sparse node mapping.
 * @param sparseToDenseLUT Holds the sparse-to-dense node mapping.
 * @param lutOffset Holds the point-LUT offset.
 * @param threshold The merging threshold.
 * @param nodeAmount Cell (node) amount of the current hierarchy.
 * @param gridSize Grid size of the current hierarchy. (e.g. 128)
 * @param lowerGridSize Grid size of one hierarchy level below. (e.g. 256)
 * @param nodeOffset The accumulated amount of dense nodes for the current hierarchy level.
 * e.g. level=128 -> nodeOffset = 512*512*512 + 256*256*256
 * @param nodeOffsetLower The accumulated amount of dense nodes of one hierarchy level below.
 * e.g. level=128 -> nodeOffsetLower = 512*512*512
 */
__global__ void kernelMergeHierarchical (
        Node* octree,
        const uint32_t* countingGrid,
        const int* denseToSparseLUT,
        uint32_t* lutOffset,
        uint32_t threshold,
        uint32_t nodeAmount,
        uint32_t gridSize,
        uint32_t lowerGridSize,
        uint32_t nodeOffset,
        uint32_t nodeOffsetLower)
{
    unsigned int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

    uint32_t denseVoxelIndex = nodeOffset + index; // May be invalid

    if (index >= nodeAmount || denseToSparseLUT[denseVoxelIndex] == -1)
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

    // Update the current node
    Node* node          = octree + sparseVoxelIndex;
    uint32_t pointCount = countingGrid[denseVoxelIndex];
    bool isFinished     = (pointCount >= threshold);

    // Update the current node
    node->pointCount = isFinished ? 0 : pointCount;
    node->isInternal = isFinished;
    node->isFinished = isFinished;

    // Assign children chunks and sum up all point in child nodes
    auto sum = 0;
#pragma unroll
    for (uint8_t i = 0; i < 8; ++i)
    {
        node->childNodes[i] = (countingGrid[childNodes[i]] > 0) ? denseToSparseLUT[childNodes[i]] : -1;
        if (node->childNodes[i] != -1)
        {
            // Update isFinished in each child
            (octree + node->childNodes[i])->isFinished = isFinished;

            // Assign current sparse chunk index to child as parentChunkIndex
            (octree + node->childNodes[i])->parentNode = sparseVoxelIndex;

            sum += (octree + node->childNodes[i])->pointCount;
        }
    }

    // ##################################################################################

    if (isFinished && sum > 0)
    {
        // Determine the start index inside the dataLUT for all children chunks
        uint32_t dataLUTIndex = atomicAdd (lutOffset, sum);
#pragma unroll
        for (int childNode : node->childNodes)
        {
            if (childNode != -1)
            {
                // 6.2. Update the exact index for the child within the dataLUT
                (octree + childNode)->dataIdx = dataLUTIndex;
                dataLUTIndex += (octree + childNode)->pointCount;
            }
        }
    }
}
} // namespace chunking