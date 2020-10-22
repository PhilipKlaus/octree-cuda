//
// Created by KlausP on 05.10.2020.
//

#include <fstream>
#include "pointcloud.h"
#include <iostream>
#include <string>
#include <assert.h>

unique_ptr<Chunk[]> PointCloud::getOctree() {
    return itsGrid->toHost();
}

unique_ptr<Vector3[]> PointCloud::getChunkData() {
    return itsChunkData->toHost();
}

void exportTreeNode(Chunk *tree, Vector3 *chunkData, uint64_t level, uint64_t index) {
    if(tree[index].isFinished && tree[index].pointCount > 0) {
        std::ofstream ply;
        ply.open (std::string("tree_" + std::to_string(level) + "_" + std::to_string(index) + "_" + std::to_string(tree[index].pointCount) + ".ply"), std::ios::binary);

        ply << "ply\n"
               "format binary_little_endian 1.0\n"
               "comment Created by AIT Austrian Institute of Technology\n"
               "element vertex "
            << tree[index].pointCount
            << "\n"
               "property float x\n"
               "property float y\n"
               "property float z\n"
               "end_header\n";
        for (uint64_t u = 0; u < tree[index].pointCount; ++u)
        {
            ply.write (reinterpret_cast<const char*> (&(chunkData[tree[index].chunkDataIndex + u].x)), sizeof (float));
            ply.write (reinterpret_cast<const char*> (&(chunkData[tree[index].chunkDataIndex + u].y)), sizeof (float));
            ply.write (reinterpret_cast<const char*> (&(chunkData[tree[index].chunkDataIndex + u].z)), sizeof (float));
        }
        ply.close ();
    }
    else {
        if (level > 0) {
            for(unsigned long long childrenChunk : tree[index].childrenChunks) {
                exportTreeNode(tree, chunkData, level - 1, childrenChunk);
            }
        }
    }
}

void PointCloud::exportGlobalTree() {
    auto octree = getOctree();
    auto chunkData = getChunkData();
    uint64_t topLevelIndex = itsCellAmount - 1;

    exportTreeNode(octree.get(), chunkData.get(), 7, topLevelIndex);
}
