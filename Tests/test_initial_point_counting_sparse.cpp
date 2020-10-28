//
// Created by KlausP on 28.10.2020.
//

#include "catch2/catch.hpp"
#include "../lib/src/tools.cuh"
#include "pointcloud.h"

TEST_CASE ("Test initial sparse point counting", "counting sparse") {

    // Create test data point cloud
    unique_ptr<CudaArray<Vector3>> cuboid = tools::generate_point_cloud_cuboid(256);

    auto cloud = make_unique<PointCloud>(move(cuboid));

    cloud->getMetadata().pointAmount = 256 * 256 * 256;
    cloud->getMetadata().boundingBox.minimum = Vector3 {0.5, 0.5, 0.5};
    cloud->getMetadata().boundingBox.maximum = Vector3 {255.5, 255.5, 255.5};
    cloud->getMetadata().cloudOffset = Vector3 {0.5, 0.5, 0.5};
    cloud->getMetadata().scale = {1.f, 1.f, 1.f};

    cloud->initialPointCountingSparse(7);

    auto denseCount = cloud->getDensePointCount();

    // Require that all cells are filled and that there is no empty space
    REQUIRE(cloud->getCellAmountSparse() == pow(128, 3));

    // Require that the sum of the accumulated point counts equaly to the actual point amount of the cloud
    uint32_t sum = 0;
    for(int i = 0; i < pow(128, 3); ++i) {
        REQUIRE(denseCount[i] == 8);
        sum += denseCount[i];
    }
    REQUIRE(sum == cloud->getMetadata().pointAmount);
}
