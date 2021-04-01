/**
 * @file octree_data.cuh
 * @author Philip Klaus
 * @brief Contains an OctreeData class
 */

#pragma once

#include "metadata.cuh"
#include "types.cuh"
#include <cstdint>
#include <vector>

class HierarchyNotCreatedException : public std::exception
{
public:
    using std::exception::exception;
};


/**
 * The OctreeData class represents and manages an octree data structure and its metadata.
 */
class OctreeData
{
public:
    OctreeData (uint32_t chunkingGrid);
    OctreeData (const OctreeData&) = delete;

    void createHierarchy (uint32_t nodeAmountSparse);

    uint32_t getRootIndex () const;
    uint32_t getNodeAmount (uint8_t level) const;
    uint32_t getGridSize (uint8_t level) const;
    uint32_t getNodeOffset (uint8_t level) const;
    const Node& getNode (uint32_t index);

    void copyToHost ();
    const std::shared_ptr<Node[]>& getHost ();
    Node* getDevice () const;

    const OctreeInfo& getNodeStatistics () const;

    void updateNodeStatistics ();

private:
    void initialize (uint32_t chunkingGrid);
    void ensureHierarchyCreated () const;
    void evaluateNodeProperties (OctreeInfo& statistics, uint32_t& pointSum, uint32_t nodeIndex, uint8_t level);
    void calculatePointVarianceInLeafNoes (float& sumVariance, uint32_t nodeIndex) const;

private:
    std::vector<uint32_t> itsNodesPerLevel;      // Holds the node amount per level (dense / bottom-up)
    std::vector<uint32_t> itsGridSizePerLevel;   // Holds the grid side length per level (bottom-up)
    std::vector<uint32_t> itsNodeOffsetperLevel; // Holds node offset per level (dense / bottom-up)

    GpuOctree itsOctree;
    std::shared_ptr<Node[]> itsOctreeHost;

    OctreeInfo itsNodeStatistics;
};

using Octree = std::unique_ptr<OctreeData>;
