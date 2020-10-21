//
// Created by KlausP on 05.10.2020.
//

#include <fstream>
#include "pointcloud.h"
#include <iostream>
#include <string>
#include <assert.h>

unique_ptr<Chunk[]> PointCloud::getCountingGrid() {
    return itsGrid->toHost();
}

unique_ptr<Vector3[]> PointCloud::getTreeData() {
    return itsChunkData->toHost();
}

void PointCloud::exportGlobalTree() {
    auto grid = getCountingGrid();
    auto treeData = getTreeData();

    uint64_t cellOffset = 0;
    uint64_t level = 0;
    for(uint64_t gridSize = itsGridBaseSideLength; gridSize > 0; gridSize >>= 1) {

        for(uint64_t i = 0; i < pow(gridSize, 3); ++i) {

            if(grid[cellOffset + i].isFinished && grid[cellOffset + i].pointCount > 0) {
                uint64_t treeIndex = grid[cellOffset + i].treeIndex;

                std::ofstream ply;
                ply.open (std::string("tree_" + std::to_string(level) + "_" + std::to_string(i) + "_" + std::to_string(grid[cellOffset + i].pointCount) + ".ply"), std::ios::binary);

                ply << "ply\n"
                       "format binary_little_endian 1.0\n"
                       "comment Created by AIT Austrian Institute of Technology\n"
                       "element vertex "
                    << grid[cellOffset + i].pointCount
                    << "\n"
                       "property float x\n"
                       "property float y\n"
                       "property float z\n"
                       "end_header\n";
                for (uint64_t u = 0; u < grid[cellOffset + i].pointCount; ++u)
                {
                    ply.write (reinterpret_cast<const char*> (&(treeData[treeIndex + u].x)), sizeof (float));
                    ply.write (reinterpret_cast<const char*> (&(treeData[treeIndex + u].y)), sizeof (float));
                    ply.write (reinterpret_cast<const char*> (&(treeData[treeIndex + u].z)), sizeof (float));
                }
                ply.close ();
            }
        }
        ++level;
        cellOffset += static_cast<uint64_t >(pow(gridSize, 3));
    }
}
