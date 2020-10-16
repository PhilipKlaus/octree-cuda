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

vector<unique_ptr<Chunk[]>> PointCloud::getCountingGrid() {
    vector<unique_ptr<Chunk[]>> output;
    for(int i = 0; i < itsGrid.size(); ++i) {
        output.push_back(itsGrid[i]->toHost());
    }
    return output;
}

unique_ptr<Vector3[]> PointCloud::getTreeData() {
    return itsTreeData->toHost();
}

void PointCloud::exportGlobalTree() {
    auto host = getCountingGrid();
    auto treeData = getTreeData();

    uint32_t level = 0;
    for(int gridSize = itsGridSize; gridSize > 1; gridSize >>= 1) {

        for(uint32_t i = 0; i < pow(gridSize, 3); ++i) {

            if(host[level][i].isFinished && host[level][i].count > 0) {
                uint32_t  treeIndex = host[level][i].treeIndex;

                std::ofstream ply;
                ply.open (std::string("tree_" + std::to_string(level) + "_" + std::to_string(i) + "_" + std::to_string(host[level][i].count) + ".ply"), std::ios::binary);

                ply << "ply\n"
                       "format binary_little_endian 1.0\n"
                       "comment Created by AIT Austrian Institute of Technology\n"
                       "element vertex "
                    << host[level][i].count
                    << "\n"
                       "property float x\n"
                       "property float y\n"
                       "property float z\n"
                       "end_header\n";
                for (uint32_t u = 0; u < host[level][i].count; ++u)
                {
                    ply.write (reinterpret_cast<const char*> (&(treeData[treeIndex + u].x)), sizeof (float));
                    ply.write (reinterpret_cast<const char*> (&(treeData[treeIndex + u].y)), sizeof (float));
                    ply.write (reinterpret_cast<const char*> (&(treeData[treeIndex + u].z)), sizeof (float));
                }
                ply.close ();
            }
        }
        ++level;
    }
}
