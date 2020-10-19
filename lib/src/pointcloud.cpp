//
// Created by KlausP on 05.10.2020.
//

#include <fstream>
#include "pointcloud.h"
#include <iostream>
#include <string>
#include <assert.h>

void PointCloud::exportToPly(const std::string& file_name) {
    std::ofstream ply;
    ply.open (file_name, std::ios::binary);
    ply << "ply\n"
           "format binary_little_endian 1.0\n"
           "comment Created by AIT Austrian Institute of Technology\n"
           "element vertex "
        << itsData->pointCount()
        << "\n"
           "property float x\n"
           "property float y\n"
           "property float z\n"
           "end_header\n";

    unique_ptr<Vector3[]> host = itsData->toHost();
    for (auto i = 0u; i < itsData->pointCount(); ++i)
    {
        // write vertex coordinates
        ply.write (reinterpret_cast<const char*> (&(host[i].x)), sizeof (float));
        ply.write (reinterpret_cast<const char*> (&(host[i].y)), sizeof (float));
        ply.write (reinterpret_cast<const char*> (&(host[i].z)), sizeof (float));
    }
    ply.close ();
}

unique_ptr<Chunk[]> PointCloud::getCountingGrid() {
    return itsGrid->toHost();
}

unique_ptr<Vector3[]> PointCloud::getTreeData() {
    return itsTreeData->toHost();
}

void PointCloud::exportGlobalTree() {
    auto host = getCountingGrid();
    auto treeData = getTreeData();

    uint64_t cellOffset = 0;
    uint64_t level = 0;
    for(uint64_t gridSize = itsGridBaseSideLength; gridSize > 0; gridSize >>= 1) {

        for(uint64_t i = 0; i < pow(gridSize, 3); ++i) {

            if(host[cellOffset + i].isFinished && host[cellOffset + i].count > 0) {
                uint64_t treeIndex = host[cellOffset + i].treeIndex;

                std::ofstream ply;
                ply.open (std::string("tree_" + std::to_string(level) + "_" + std::to_string(i) + "_" + std::to_string(host[cellOffset + i].count) + ".ply"), std::ios::binary);

                ply << "ply\n"
                       "format binary_little_endian 1.0\n"
                       "comment Created by AIT Austrian Institute of Technology\n"
                       "element vertex "
                    << host[cellOffset + i].count
                    << "\n"
                       "property float x\n"
                       "property float y\n"
                       "property float z\n"
                       "end_header\n";
                for (uint64_t u = 0; u < host[cellOffset + i].count; ++u)
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
