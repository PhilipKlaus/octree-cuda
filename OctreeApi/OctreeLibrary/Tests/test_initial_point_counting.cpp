//
// Created by KlausP on 27.10.2020.
//

#include "catch2/catch.hpp"
#include "../src/tools.cuh"
#include "../src/defines.cuh"
#include "pointcloud.h"


TEST_CASE ("Test initial point counting", "counting") {

    // Create test data point cloud
    unique_ptr<CudaArray<Vector3>> cuboid = tools::generate_point_cloud_cuboid(256);

    auto cloud = make_unique<PointCloud>(move(cuboid));

    cloud->getMetadata().pointAmount = 256 * 256 * 256;
    cloud->getMetadata().boundingBox.minimum = Vector3 {0.5, 0.5, 0.5};
    cloud->getMetadata().boundingBox.maximum = Vector3 {255.5, 255.5, 255.5};
    cloud->getMetadata().cloudOffset = Vector3 {0.5, 0.5, 0.5};
    cloud->getMetadata().scale = {1.f, 1.f, 1.f};

    cloud->initialPointCounting(7);

    // Test if each point fall exactly in one cell
    auto octree = cloud->getOctree();

    for(int i = 0; i < pow(128, 3); ++i) {
        REQUIRE(octree[i].pointCount == 8);
        REQUIRE(octree[i].isFinished == false);
        REQUIRE(octree[i].parentChunkIndex != INVALID_INDEX);
    }
}
