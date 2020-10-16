#include <iostream>
#include <fstream>


#include "pointcloud.h"



using namespace std;

constexpr unsigned int GRID_SIZE = 128;

int main() {

#ifndef NDEBUG
    spdlog::set_level(spdlog::level::debug);
#else
    spdlog::set_level(spdlog::level::info);
#endif

    uint32_t pointAmount = 1612868;//327323;
    ifstream ifs("doom_vertices.ply", ios::binary|ios::ate);
    ifstream::pos_type pos = ifs.tellg();
    int length = pos;
    auto *pChars = new uint8_t[length];
    ifs.seekg(0, ios::beg);
    ifs.read(reinterpret_cast<char *>(pChars), length);
    ifs.close();

    Vector3 minimum {INFINITY, INFINITY, INFINITY};
    Vector3 maximum {-INFINITY, -INFINITY, -INFINITY};
    auto *points = reinterpret_cast<Vector3*>(pChars);
    for(int i = 0; i < pointAmount; ++i) {
        minimum.x = fmin(minimum.x, points[i].x);
        minimum.y = fmin(minimum.y, points[i].y);
        minimum.z = fmin(minimum.z, points[i].z);
        maximum.x = fmax(maximum.x, points[i].x);
        maximum.y = fmax(maximum.y, points[i].y);
        maximum.z = fmax(maximum.z, points[i].z);
    }

    auto data = make_unique<CudaArray<Vector3>>(pointAmount);
    data->toGPU(pChars);

    auto cloud = make_unique<PointCloud>(move(data));
    BoundingBox boundingBox{
            minimum,
            maximum
    };
    PointCloudMetadata metadata {
            pointAmount,
            boundingBox,
            minimum
    };
    cloud->initialPointCounting(7, metadata);
    cloud->performCellMerging(30000);
    cloud->distributePoints();
    cloud->exportGlobalTree();

    delete[] pChars;

}