#include "octree.cuh"
#include "time_tracker.cuh"
#include "tools.cuh"

OctreeData::OctreeData (uint32_t chunkingGrid) : itsNodeStatistics ({})
{
    itsNodeStatistics.nodeAmountDense = 0;
    itsNodeStatistics.depth           = tools::getOctreeLevel (chunkingGrid);

    initialize (chunkingGrid);
}

void OctreeData::initialize (uint32_t chunkingGrid)
{
    for (uint32_t gridSize = chunkingGrid; gridSize > 0; gridSize >>= 1)
    {
        itsGridSizePerLevel.push_back (gridSize);
        itsNodeOffsetperLevel.push_back (itsNodeStatistics.nodeAmountDense);
        itsNodesPerLevel.push_back (static_cast<uint32_t> (pow (gridSize, 3)));

        itsNodeStatistics.nodeAmountDense += static_cast<uint32_t> (pow (gridSize, 3));
    }
}

uint32_t OctreeData::getNodeAmount (uint8_t level) const
{
    return itsNodesPerLevel[level];
}

uint32_t OctreeData::getGridSize (uint8_t level) const
{
    return itsGridSizePerLevel[level];
}

uint32_t OctreeData::getNodeOffset (uint8_t level) const
{
    return itsNodeOffsetperLevel[level];
}

void OctreeData::createHierarchy (uint32_t nodeAmountSparse)
{
    itsNodeStatistics.nodeAmountSparse = nodeAmountSparse;
    itsOctree                          = createGpuOctree (nodeAmountSparse, "octreeSparse");
}

void OctreeData::copyToHost ()
{
    ensureHierarchyCreated ();
    itsOctreeHost = itsOctree->toHost ();
}

const std::shared_ptr<Node[]>& OctreeData::getHost ()
{
    ensureHierarchyCreated ();
    if (!itsOctreeHost)
    {
        copyToHost ();
    }
    return itsOctreeHost;
}

Node* OctreeData::getDevice () const
{
    ensureHierarchyCreated ();
    return itsOctree->devicePointer ();
}

const Node& OctreeData::getNode (uint32_t index)
{
    ensureHierarchyCreated ();
    return getHost ()[index];
}

const OctreeInfo& OctreeData::getNodeStatistics () const
{
    return itsNodeStatistics;
}

void OctreeData::updateNodeStatistics ()
{
    ensureHierarchyCreated ();

    // Reset Octree statistics
    itsNodeStatistics.maxLeafDepth           = 0;
    itsNodeStatistics.leafNodeAmount         = 0;
    itsNodeStatistics.internalNodeAmount       = 0;
    itsNodeStatistics.meanPointsPerLeafNode  = 0.f;
    itsNodeStatistics.stdevPointsPerLeafNode = 0.f;
    itsNodeStatistics.minPointsPerNode       = std::numeric_limits<uint32_t>::max ();
    itsNodeStatistics.maxPointsPerNode       = std::numeric_limits<uint32_t>::min ();

    uint32_t pointSum = 0;
    float sumVariance = 0.f;

    getHost ();
    evaluateNodeProperties (itsNodeStatistics, pointSum, getRootIndex (), 0);

    itsNodeStatistics.meanPointsPerLeafNode = static_cast<float> (pointSum) / itsNodeStatistics.leafNodeAmount;

    calculatePointVarianceInLeafNoes (sumVariance, getRootIndex ());
    itsNodeStatistics.stdevPointsPerLeafNode = sqrt (sumVariance / itsNodeStatistics.leafNodeAmount);
}

void OctreeData::evaluateNodeProperties (OctreeInfo& statistics, uint32_t& pointSum, uint32_t nodeIndex, uint8_t level)
{
    Node node = itsOctreeHost[nodeIndex];

    // Leaf node
    if (!node.isInternal)
    {
        statistics.maxLeafDepth = max (statistics.maxLeafDepth, level);
        statistics.leafNodeAmount += 1;
        pointSum += node.pointCount;
        statistics.minPointsPerNode =
                node.pointCount < statistics.minPointsPerNode ? node.pointCount : statistics.minPointsPerNode;
        statistics.maxPointsPerNode =
                node.pointCount > statistics.maxPointsPerNode ? node.pointCount : statistics.maxPointsPerNode;
    }

    // Parent node
    else
    {
        statistics.internalNodeAmount += 1;
        for (int childNode : node.childNodes)
        {
            if (childNode != -1)
            {
                evaluateNodeProperties (statistics, pointSum, childNode, level + 1);
            }
        }
    }
}

uint32_t OctreeData::getRootIndex () const
{
    ensureHierarchyCreated ();
    return itsNodeStatistics.nodeAmountSparse - 1;
}

void OctreeData::calculatePointVarianceInLeafNoes (float& sumVariance, uint32_t nodeIndex) const
{
    Node node = itsOctreeHost[nodeIndex];

    // Leaf node
    if (!node.isInternal)
    {
        sumVariance += pow (static_cast<float> (node.pointCount) - itsNodeStatistics.meanPointsPerLeafNode, 2.f);
    }

    // Parent node
    else
    {
        for (int childIndex : node.childNodes)
        {
            if (childIndex != -1)
            {
                calculatePointVarianceInLeafNoes (sumVariance, childIndex);
            }
        }
    }
}

void OctreeData::ensureHierarchyCreated () const
{
    if (!itsOctree)
    {
        throw HierarchyNotCreatedException ();
    }
}
