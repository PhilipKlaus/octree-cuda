// Standard library
#include <iostream>
#include <fstream>

// Local dependencies
#include "pointcloud.h"
#include "eventWatcher.h"

using namespace std;


int main() {

#ifndef NDEBUG
    spdlog::set_level(spdlog::level::debug);
#else
    spdlog::set_level(spdlog::level::info);
#endif
    EventWatcher& watcher = EventWatcher::getInstance();
    watcher.reservedMemoryEvent(0, "Program start");

    // Hardcoded: read PLY: ToDo: Include PLY library
    uint32_t pointAmount = 1612868;//327323;
    ifstream ifs("doom_vertices.ply", ios::binary|ios::ate);
    ifstream::pos_type pos = ifs.tellg();
    std::streamoff length = pos;
    auto *pChars = new uint8_t[length];
    ifs.seekg(0, ios::beg);
    ifs.read(reinterpret_cast<char *>(pChars), length);
    ifs.close();

    // Calculate bounding box: ToDo: Read from LAZ/PLY library
    Vector3 minimum {INFINITY, INFINITY, INFINITY};
    Vector3 maximum {-INFINITY, -INFINITY, -INFINITY};
    auto *points = reinterpret_cast<Vector3*>(pChars);
    for(uint32_t i = 0; i < pointAmount; ++i) {
        minimum.x = fmin(minimum.x, points[i].x);
        minimum.y = fmin(minimum.y, points[i].y);
        minimum.z = fmin(minimum.z, points[i].z);
        maximum.x = fmax(maximum.x, points[i].x);
        maximum.y = fmax(maximum.y, points[i].y);
        maximum.z = fmax(maximum.z, points[i].z);
    }

    // Copy ply point cloud to GPU
    auto data = make_unique<CudaArray<Vector3>>(pointAmount, "pointcloud");
    data->toGPU(pChars);

    // Set cloud settings
    auto cloud = make_unique<PointCloud>(move(data));
    cloud->getMetadata().pointAmount = pointAmount;
    cloud->getMetadata().boundingBox.minimum = minimum;
    cloud->getMetadata().boundingBox.maximum = maximum;
    cloud->getMetadata().cloudOffset = minimum;
    cloud->getMetadata().scale = {1.f, 1.f, 1.f};

    // Perform subsampling
    cloud->initialPointCountingSparse(7);
    cloud->performCellMergingSparse(30000);
    cloud->distributePointsSparse();
    cloud->exportOctreeSparse(points);
    //cloud->exportOctree(points);
    //cloud->exportTimeMeasurement();
    //cloud->distributePointsSparse();
    /*cloud->initialPointCounting(7);
    cloud->performCellMerging(30000);
    cloud->distributePoints();
    cloud->exportOctree(points);
    cloud->exportTimeMeasurement();*/
    delete[] pChars;

}