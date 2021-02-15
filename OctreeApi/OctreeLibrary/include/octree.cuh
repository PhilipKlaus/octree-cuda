/**
 * @file octree.cuh
 * @author Philip Klaus
 * @brief Contains the Octree class for wrapping octree related data and methods
 */
#include <cstdint>
#include "types.cuh"


class Octree
{
public:
    explicit Octree (uint32_t nodeAmountSparse)
    {
        itsOctree = createGpuOctree (nodeAmountSparse, "octreeSparse");
    }

    Octree(const Octree&) = delete;

    void copyToHost() {
        itsOctreeHost = itsOctree->toHost();
    }

    const std::shared_ptr<Chunk[]>& host() {
        if(!itsOctreeHost) {
            copyToHost();
        }
        return itsOctreeHost;
    }
private:
    GpuOctree itsOctree;
    std::shared_ptr<Chunk[]> itsOctreeHost;
};