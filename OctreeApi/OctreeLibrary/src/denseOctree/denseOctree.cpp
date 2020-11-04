//
// Created by KlausP on 04.11.2020.
//

#include <denseOctree.h>

uint32_t DenseOctree::exportTreeNode(
        Vector3* cpuPointCloud,
        const unique_ptr<Chunk[]> &octree,
        const unique_ptr<uint32_t[]> &dataLUT,
        uint32_t level,
        uint32_t index,
        const string &folderPath
        ) {
    uint32_t count = 0;

    if(octree[index].isFinished && octree[index].pointCount > 0) {
        count = octree[index].pointCount;
        std::ofstream ply;
        ply.open (
                folderPath +
                "\\tree_" + std::to_string(level) +
                "_" +
                std::to_string(index) +
                "_" +
                std::to_string(octree[index].pointCount) +
                ".ply",
                std::ios::binary
        );

        ply << "ply\n"
               "format binary_little_endian 1.0\n"
               "comment Created by AIT Austrian Institute of Technology\n"
               "element vertex "
            << octree[index].pointCount
            << "\n"
               "property float x\n"
               "property float y\n"
               "property float z\n"
               "end_header\n";
        for (uint32_t u = 0; u < octree[index].pointCount; ++u)
        {
            ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[dataLUT[octree[index].chunkDataIndex + u]].x)), sizeof (float));
            ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[dataLUT[octree[index].chunkDataIndex + u]].y)), sizeof (float));
            ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[dataLUT[octree[index].chunkDataIndex + u]].z)), sizeof (float));
        }
        ply.close ();
    }
    else {
        if (level > 0) {
            for(uint32_t childrenChunk : octree[index].childrenChunks) {
                count += exportTreeNode(cpuPointCloud, octree, dataLUT, level - 1, childrenChunk, folderPath);
            }
        }
    }
    return count;
}

void DenseOctree::exportOctree(const string &folderPath) {
    auto cpuPointCloud = itsCloudData->toHost();
    auto octree = itsOctreeDense->toHost();
    auto dataLUT = itsDataLUT->toHost();
    uint32_t topLevelIndex = itsVoxelAmountDense - 1;
    uint32_t exportedPoints = exportTreeNode(cpuPointCloud.get(), octree, dataLUT, itsGlobalOctreeDepth, topLevelIndex, folderPath); // ToDo: Remove hard-coded level
    assert(exportedPoints == itsMetadata.pointAmount);
}

void DenseOctree::freeGpuMemory() {
    itsOctreeDense.reset();
    spdlog::debug("Dense octree GPU memory deleted");
}

unique_ptr<Chunk[]> DenseOctree::getOctreeDense() {
    return itsOctreeDense->toHost();
}
