#include "octree_data.cuh"
#include "tools.cuh"

OctreeData::OctreeData (uint32_t chunkingGrid) : itsDepth (0), itsChunkingGrid (chunkingGrid), itsNodeAmountDense (0)
{
    initialize ();
}

void OctreeData::initialize ()
{
    itsDepth = tools::getOctreeLevel (itsChunkingGrid);

    for (uint32_t gridSize = itsChunkingGrid; gridSize > 0; gridSize >>= 1)
    {
        itsGridSizePerLevel.push_back (gridSize);
        itsNodeOffsetperLevel.push_back (itsNodeAmountDense);
        itsNodesPerLevel.push_back (static_cast<uint32_t> (pow (gridSize, 3)));
        itsNodeAmountDense += static_cast<uint32_t> (pow (gridSize, 3));
    }
}
uint8_t OctreeData::getDepth ()
{
    return itsDepth;
}
uint32_t OctreeData::getNodes (uint8_t level)
{
    return itsNodesPerLevel[level];
}
uint32_t OctreeData::getGridSize (uint8_t level)
{
    return itsGridSizePerLevel[level];
}
uint32_t OctreeData::getNodeOffset (uint8_t level)
{
    return itsNodeOffsetperLevel[level];
}
uint32_t OctreeData::getOverallNodes ()
{
    return itsNodeAmountDense;
}
void OctreeData::createOctree (uint32_t nodeAmountSparse)
{
    itsOctree = createGpuOctree (nodeAmountSparse, "octreeSparse");
}
void OctreeData::copyToHost ()
{
    auto start = std::chrono::high_resolution_clock::now ();
    itsOctreeHost = itsOctree->toHost();
    auto finish                           = std::chrono::high_resolution_clock::now ();
    std::chrono::duration<double> elapsed = finish - start;
    spdlog::info("Copied octree from device to host in: {}s", elapsed.count());
}
const std::shared_ptr<Chunk[]>& OctreeData::getHost ()
{
    if(!itsOctreeHost) {
        copyToHost();
    }
    return itsOctreeHost;
}
Chunk* OctreeData::getDevice ()
{
    return itsOctree->devicePointer();
}
