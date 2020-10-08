#include <iostream>
#include <fstream>

#include "tools.cuh"
#include "pointcloud.h"

using namespace std;

constexpr unsigned int GRID_SIZE = 128;

int main() {

    // Create equally spaced point cloud cuboid
    //unsigned int elementsPerCuboidSide = 128;
    //unique_ptr<CudaArray<Vector3>> cuboid = generate_point_cloud_cuboid(elementsPerCuboidSide);

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

    cout << "min: x: " << minimum.x << ", y: " << minimum.y << ", z: " << minimum.z << endl;
    cout << "max: x: " << maximum.x << ", y: " << maximum.y << ", z: " << maximum.z << endl;
    cout << "width: " << maximum.x - minimum.x << endl;
    cout << "height: " << maximum.y - minimum.y << endl;
    cout << "depth: " << maximum.z - minimum.z << endl;

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

    auto grid = cloud->getCountingGrid();
    cloud->performCellMerging(300000);
    uint32_t sum = 0;
    for(int i = 0; i < pow(128,3); ++i) {
        sum += grid[0][i].count;
    }
    cout << "sum: " << sum << endl;

    grid = cloud->getCountingGrid();
    uint32_t level = 0;
    sum = 0;
    for(int gridSize = 128; gridSize > 1; gridSize >>= 1) {
        for(uint32_t i = 0; i < pow(gridSize, 3); ++i)
        {
            if(grid[level][i].isFinished)
                sum += grid[level][i].count;
        }

        ++level;
    }
    cout << "sum: " << sum << endl;
    cloud->distributePoints();

    grid = cloud->getCountingGrid();
    level = 0;
    sum = 0;
    for(int gridSize = 128; gridSize > 1; gridSize >>= 1) {
        for(uint32_t i = 0; i < pow(gridSize, 3); ++i)
        {
            if(grid[level][i].isFinished)
                sum += grid[level][i].count;
        }

        ++level;
    }
    cout << "sum: " << sum << endl;
    cloud->exportGlobalTree();

    delete[] pChars;

}