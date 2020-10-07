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
    std::cout << itsData->pointCount() << std::endl;
    ply << "ply\n"
           "format binary_little_endian 1.0\n"
           "comment Created by AIT Austrian Institute of Technology\n"
           "element vertex "
        << itsData->pointCount()
        << "\n"
           "property double x\n"
           "property double y\n"
           "property double z\n"
           "end_header\n";

    unique_ptr<Vector3[]> host = itsData->toHost();
    for (auto i = 0u; i < itsData->pointCount(); ++i)
    {
        // write vertex coordinates
        ply.write (reinterpret_cast<const char*> (&(host[i].x)), sizeof (double));
        ply.write (reinterpret_cast<const char*> (&(host[i].y)), sizeof (double));
        ply.write (reinterpret_cast<const char*> (&(host[i].z)), sizeof (double));
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

        uint32_t sum = 0;
        for(uint32_t i = 0; i < pow(gridSize, 3); ++i) {
            if (host[level][i].isFinished) {
                sum += host[level][i].count;
            }
        }

        std::ofstream ply;
        ply.open (std::string("tree_" + std::to_string(level) + ".ply"), std::ios::binary);

        ply << "ply\n"
               "format binary_little_endian 1.0\n"
               "comment Created by AIT Austrian Institute of Technology\n"
               "element vertex "
            << sum
            << "\n"
               "property double x\n"
               "property double y\n"
               "property double z\n"
               "end_header\n";


        for(uint32_t i = 0; i < pow(gridSize, 3); ++i) {

            if(host[level][i].isFinished) {
                sum += host[level][i].count;

                uint32_t  treeIndex = host[level][i].treeIndex;

                for (uint32_t u = 0; u < host[level][i].count; ++u)
                {
                    // write vertex coordinates
                    ply.write (reinterpret_cast<const char*> (&(treeData[treeIndex + u].x)), sizeof (double));
                    ply.write (reinterpret_cast<const char*> (&(treeData[treeIndex + u].y)), sizeof (double));
                    ply.write (reinterpret_cast<const char*> (&(treeData[treeIndex + u].z)), sizeof (double));
                }
            }
        }
        
        ply.close ();
        ++level;
    }
}
