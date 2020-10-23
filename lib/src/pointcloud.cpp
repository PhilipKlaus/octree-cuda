//
// Created by KlausP on 05.10.2020.
//

#include <fstream>
#include "pointcloud.h"
#include <string>

unique_ptr<Chunk[]> PointCloud::getOctree() {
    return itsOctree->toHost();
}

unique_ptr<Vector3[]> PointCloud::getChunkData() {
    return itsChunkData->toHost();
}

void PointCloud::exportTreeNode(const unique_ptr<Chunk[]> &octree, const unique_ptr<Vector3[]> &chunkData, uint64_t level, uint64_t index) {
    if(octree[index].isFinished && octree[index].pointCount > 0) {
        std::ofstream ply;
        ply.open (
                "tree_" + std::to_string(level) +
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
        for (uint64_t u = 0; u < octree[index].pointCount; ++u)
        {
            ply.write (reinterpret_cast<const char*> (&(chunkData[octree[index].chunkDataIndex + u].x)), sizeof (float));
            ply.write (reinterpret_cast<const char*> (&(chunkData[octree[index].chunkDataIndex + u].y)), sizeof (float));
            ply.write (reinterpret_cast<const char*> (&(chunkData[octree[index].chunkDataIndex + u].z)), sizeof (float));
        }
        ply.close ();
    }
    else {
        if (level > 0) {
            for(unsigned long long childrenChunk : octree[index].childrenChunks) {
                exportTreeNode(octree, chunkData, level - 1, childrenChunk);
            }
        }
    }
}

void PointCloud::exportOctree() {
    auto octree = getOctree();
    auto chunkData = getChunkData();
    uint64_t topLevelIndex = itsCellAmount - 1;

    exportTreeNode(octree, chunkData, 7, topLevelIndex);
}
