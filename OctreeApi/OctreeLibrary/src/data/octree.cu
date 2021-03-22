#include "octree.cuh"
#include "time_tracker.cuh"
#include "tools.cuh"

Octree::Octree (uint32_t chunkingGrid, uint32_t mergingThreshold) : itsMetadata ({}), itsNodeStatistics ({})
{
    itsMetadata.nodeAmountDense  = 0;
    itsMetadata.mergingThreshold = mergingThreshold;
    itsMetadata.chunkingGrid     = chunkingGrid;
    itsMetadata.depth            = tools::getOctreeLevel (chunkingGrid);
    initialize ();
}

void Octree::initialize ()
{
    for (uint32_t gridSize = itsMetadata.chunkingGrid; gridSize > 0; gridSize >>= 1)
    {
        itsGridSizePerLevel.push_back (gridSize);
        itsNodeOffsetperLevel.push_back (itsMetadata.nodeAmountDense);
        itsNodesPerLevel.push_back (static_cast<uint32_t> (pow (gridSize, 3)));
        itsMetadata.nodeAmountDense += static_cast<uint32_t> (pow (gridSize, 3));
    }
}

uint32_t Octree::getNodeAmount (uint8_t level) const
{
    return itsNodesPerLevel[level];
}

uint32_t Octree::getGridSize (uint8_t level) const
{
    return itsGridSizePerLevel[level];
}

uint32_t Octree::getNodeOffset (uint8_t level) const
{
    return itsNodeOffsetperLevel[level];
}

void Octree::createHierarchy (uint32_t nodeAmountSparse)
{
    itsMetadata.nodeAmountSparse = nodeAmountSparse;
    itsOctree                    = createGpuOctree (nodeAmountSparse, "octreeSparse");
}

void Octree::copyToHost ()
{
    ensureHierarchyCreated();
    itsOctreeHost = itsOctree->toHost ();
}

const std::shared_ptr<Chunk[]>& Octree::getHost ()
{
    ensureHierarchyCreated();
    if (!itsOctreeHost)
    {
        copyToHost ();
    }
    return itsOctreeHost;
}

Chunk* Octree::getDevice () const
{
    ensureHierarchyCreated();
    return itsOctree->devicePointer ();
}

const Chunk& Octree::getNode (uint32_t index)
{
    ensureHierarchyCreated();
    return getHost ()[index];
}

const OctreeMetadata& Octree::getMetadata () const
{
    return itsMetadata;
}

const NodeStatistics& Octree::getNodeStatistics () const
{
    return itsNodeStatistics;
}

void Octree::updateNodeStatistics ()
{
    ensureHierarchyCreated();

    // Reset Octree statistics
    itsNodeStatistics.leafNodeAmount         = 0;
    itsNodeStatistics.parentNodeAmount       = 0;
    itsNodeStatistics.meanPointsPerLeafNode  = 0.f;
    itsNodeStatistics.stdevPointsPerLeafNode = 0.f;
    itsNodeStatistics.minPointsPerNode       = std::numeric_limits<uint32_t>::max ();
    itsNodeStatistics.maxPointsPerNode       = std::numeric_limits<uint32_t>::min ();

    uint32_t pointSum = 0;
    float sumVariance = 0.f;

    getHost ();
    evaluateNodeProperties (itsNodeStatistics, pointSum, getRootIndex ());

    itsNodeStatistics.meanPointsPerLeafNode = static_cast<float> (pointSum) / itsNodeStatistics.leafNodeAmount;

    calculatePointVarianceInLeafNoes (sumVariance, getRootIndex ());
    itsNodeStatistics.stdevPointsPerLeafNode = sqrt (sumVariance / itsNodeStatistics.leafNodeAmount);
}

void Octree::evaluateNodeProperties (NodeStatistics& statistics, uint32_t& pointSum, uint32_t nodeIndex)
{
    Chunk chunk = itsOctreeHost[nodeIndex];

    // Leaf node
    if (!chunk.isParent)
    {
        statistics.leafNodeAmount += 1;
        pointSum += chunk.pointCount;
        statistics.minPointsPerNode =
                chunk.pointCount < statistics.minPointsPerNode ? chunk.pointCount : statistics.minPointsPerNode;
        statistics.maxPointsPerNode =
                chunk.pointCount > statistics.maxPointsPerNode ? chunk.pointCount : statistics.maxPointsPerNode;
    }

    // Parent node
    else
    {
        statistics.parentNodeAmount += 1;
        for (int childrenChunk : chunk.childrenChunks)
        {
            if (childrenChunk != -1)
            {
                evaluateNodeProperties (statistics, pointSum, childrenChunk);
            }
        }
    }
}

uint32_t Octree::getRootIndex () const
{
    ensureHierarchyCreated();
    return itsMetadata.nodeAmountSparse - 1;
}

void Octree::calculatePointVarianceInLeafNoes (float& sumVariance, uint32_t nodeIndex) const
{
    Chunk chunk = itsOctreeHost[nodeIndex];

    // Leaf node
    if (!chunk.isParent)
    {
        sumVariance += pow (static_cast<float> (chunk.pointCount) - itsNodeStatistics.meanPointsPerLeafNode, 2.f);
    }

    // Parent node
    else
    {
        for (int childIndex : chunk.childrenChunks)
        {
            if (childIndex != -1)
            {
                calculatePointVarianceInLeafNoes (sumVariance, childIndex);
            }
        }
    }
}

void Octree::ensureHierarchyCreated () const
{
    if(!itsOctree) {
        throw HierarchyNotCreatedException();
    }
}
