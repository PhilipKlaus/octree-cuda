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

uint32_t PointCloud::exportTreeNode(const unique_ptr<Chunk[]> &octree, const unique_ptr<Vector3[]> &chunkData, uint32_t level, uint32_t index) {
    uint32_t count = 0;

    if(octree[index].isFinished && octree[index].pointCount > 0) {
        count = octree[index].pointCount;
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
        for (uint32_t u = 0; u < octree[index].pointCount; ++u)
        {
            ply.write (reinterpret_cast<const char*> (&(chunkData[octree[index].chunkDataIndex + u].x)), sizeof (float));
            ply.write (reinterpret_cast<const char*> (&(chunkData[octree[index].chunkDataIndex + u].y)), sizeof (float));
            ply.write (reinterpret_cast<const char*> (&(chunkData[octree[index].chunkDataIndex + u].z)), sizeof (float));
        }
        ply.close ();
    }
    else {
        if (level > 0) {
            for(uint32_t childrenChunk : octree[index].childrenChunks) {
                count += exportTreeNode(octree, chunkData, level - 1, childrenChunk);
            }
        }
    }
    return count;
}

void PointCloud::exportOctree() {
    auto octree = getOctree();
    auto chunkData = getChunkData();
    uint32_t topLevelIndex = itsCellAmount - 1;
    uint32_t exportedPoints = exportTreeNode(octree, chunkData, 7, topLevelIndex); // ToDo: Remove hard-coded level
    assert(exportedPoints == itsMetadata.pointAmount);
}

void PointCloud::exportTimeMeasurement() {
    std::string timings = to_string(itsInitialPointCountTime);
    std::ofstream timingCsv;
    timingCsv.open ("timings.csv", std::ios::out);
    timingCsv << "initialPointCounting";
    uint32_t i = 0;
    for(float timing : itsMerginTime) {
        timingCsv << ",merging_" << (itsGridBaseSideLength >> i++);
        timings += ("," + std::to_string(timing));
    }
    timingCsv << ",distribution" << std::endl;
    timings += ("," + to_string(itsDistributionTime));
    timingCsv << timings;
    timingCsv.close();
}
