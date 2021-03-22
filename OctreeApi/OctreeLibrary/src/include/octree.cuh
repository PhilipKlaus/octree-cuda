/**
 * @file octree_data.cuh
 * @author Philip Klaus
 * @brief Contains an Octree class
 */

#pragma once

#include "metadata.cuh"
#include "types.cuh"
#include <cstdint>
#include <vector>

/**
 * The Octree class represents and manages an octree data structure and its metadata.
 */
class Octree
{
public:
    Octree (uint32_t chunkingGrid, uint32_t mergingThreshold);
    Octree (const Octree&) = delete;

    void createHierarchy (uint32_t nodeAmountSparse);

    uint32_t getRootIndex () const;
    uint32_t getNodeAmount (uint8_t level) const;
    uint32_t getGridSize (uint8_t level) const;
    uint32_t getNodeOffset (uint8_t level) const;
    const Chunk& getNode (uint32_t index);

    void copyToHost ();
    const std::shared_ptr<Chunk[]>& getHost ();
    Chunk* getDevice () const;

    const OctreeMetadata& getMetadata () const;
    const NodeStatistics& getNodeStatistics () const;

    void updateNodeStatistics ();

private:
    void initialize ();
    void evaluateNodeProperties (NodeStatistics& statistics, uint32_t& pointSum, uint32_t nodeIndex);
    void calculatePointVarianceInLeafNoes (float& sumVariance, uint32_t nodeIndex) const;

private:
    std::vector<uint32_t> itsNodesPerLevel;      // Holds the node amount per level (dense / bottom-up)
    std::vector<uint32_t> itsGridSizePerLevel;   // Holds the grid side length per level (bottom-up)
    std::vector<uint32_t> itsNodeOffsetperLevel; // Holds node offset per level (dense / bottom-up)

    GpuOctree itsOctree;
    std::shared_ptr<Chunk[]> itsOctreeHost;

    NodeStatistics itsNodeStatistics;
    OctreeMetadata itsMetadata;
};
