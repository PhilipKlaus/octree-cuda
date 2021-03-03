/**
 * @file octree_data.cuh
 * @author Philip Klaus
 * @brief Contains OctreeData, a class for wrapping octree related functionality and data
 */

#pragma once

#include "types.cuh"
#include <cstdint>
#include <vector>

class Octree
{
public:
    Octree (uint32_t chunkingGrid);
    Octree (const Octree&) = delete;

    void createOctree (uint32_t nodeAmountSparse);
    void copyToHost ();

    uint8_t getDepth ();
    uint32_t getNodes (uint8_t level);
    uint32_t getGridSize (uint8_t level);
    uint32_t getNodeOffset (uint8_t level);
    uint32_t getOverallNodes ();

    const std::shared_ptr<Chunk[]>& getHost ();
    Chunk* getDevice ();

    const Chunk& getNode (uint32_t index);

private:
    void initialize ();

private:
    uint32_t itsDepth;
    uint32_t itsChunkingGrid;
    uint32_t itsNodeAmountDense;
    std::vector<uint32_t> itsNodesPerLevel;      // Holds the node amount per level (dense / bottom-up)
    std::vector<uint32_t> itsGridSizePerLevel;   // Holds the grid side length per level (bottom-up)
    std::vector<uint32_t> itsNodeOffsetperLevel; // Holds node offset per level (dense / bottom-up)

    GpuOctree itsOctree;
    std::shared_ptr<Chunk[]> itsOctreeHost;
};