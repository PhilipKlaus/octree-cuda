#include <iostream>

#include "tools.cuh"
#include "pointcloud.h"

using namespace std;

constexpr unsigned int GRID_SIZE = 128;

int main() {

    // Create equally spaced point cloud cuboid
    unsigned int elementsPerCuboidSide = 128;
    unique_ptr<CudaArray<Vector3>> cuboid = generate_point_cloud_cuboid(elementsPerCuboidSide);

    auto cloud = make_unique<PointCloud>(move(cuboid));

    BoundingBox boundingBox{
            Vector3 {0.5, 0.5, 0.5},
            Vector3 {127.5, 127.5, 127.5}
    };
    PointCloudMetadata metadata {
            500 * 500 * 500,
            boundingBox,
            {0.5, 0.5, 0.5}
    };
    cloud->initialPointCounting(7, metadata);
    cloud->performCellMerging();
    cloud->distributePoints();
    cloud->exportGlobalTree();
    /*auto host = cloud->getCountingGrid();
    auto treeData = cloud->getTreeData();

    for(int i = 0; i < pow(8, 3); ++i) {
        uint32_t index = host[4][i].treeIndex;
        for(int u = 0; u < host[4][i].count; ++u) {
            cout << "x: " << treeData[index + u].x << " y: " << treeData[index + u].y << " z: " << treeData[index + u].z << endl;
        }
    }
*/
}