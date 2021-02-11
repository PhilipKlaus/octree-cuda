//
// Created by KlausP on 09.02.2021.
//

#pragma once

#include <cstdint>
#include <vector>

class OctreeData
{
public:
    OctreeData (uint32_t chunkingGrid);
    uint8_t getDepth ();
    uint32_t getNodes (uint8_t level);
    uint32_t getGridSize (uint8_t level);
    uint32_t getNodeOffset (uint8_t level);
    uint32_t getOverallNodes ();

private:
    void initialize ();

private:
    uint32_t itsDepth;
    uint32_t itsChunkingGrid;
    uint32_t itsNodeAmountDense;
    std::vector<uint32_t> itsNodesPerLevel;      // Holds the node amount per level (dense / bottom-up)
    std::vector<uint32_t> itsGridSizePerLevel;   // Holds the grid side length per level (bottom-up)
    std::vector<uint32_t> itsNodeOffsetperLevel; // Holds node offset per level (dense / bottom-up)
};
