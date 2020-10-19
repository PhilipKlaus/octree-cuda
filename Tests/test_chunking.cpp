#include "catch2/catch.hpp"
#include "../lib/src/tools.cuh"
#include "pointcloud.h"


TEST_CASE ("Test initial point counting and merging ", "[counting + merging]") {

    // Create test data point cloud
    unique_ptr<CudaArray<Vector3>> cuboid = tools::generate_point_cloud_cuboid(128);

    auto cloud = make_unique<PointCloud>(move(cuboid));

    cloud->getMetadata().pointAmount = 128 * 128 * 128;
    cloud->getMetadata().boundingBox.minimum = Vector3 {0.5, 0.5, 0.5};
    cloud->getMetadata().boundingBox.maximum = Vector3 {127.5, 127.5, 127.5};
    cloud->getMetadata().cloudOffset = Vector3 {0.5, 0.5, 0.5};
    cloud->getMetadata().scale = {1.f, 1.f, 1.f};

    cloud->initialPointCounting(7);
    cloud->performCellMerging(10000);

    // Test if each point fall exactly in one cell
    auto host = cloud->getCountingGrid();
    uint64_t cellOffset = 0;

    for(int i = 0; i < pow(128, 3); ++i) {
        REQUIRE(host[i].count == 1);
        REQUIRE(host[i].isFinished == false);
        REQUIRE(host[i].dst != nullptr);
    }
    cellOffset += static_cast<uint64_t >(pow(128, 3));

    for(int i = 0; i < pow(64, 3); ++i) {
        REQUIRE(host[cellOffset + i].count == 8);
        REQUIRE(host[cellOffset + i].isFinished == false);
        REQUIRE(host[cellOffset + i].dst != nullptr);
    }
    cellOffset += static_cast<uint64_t >(pow(64, 3));

    for(int i = 0; i < pow(32, 3); ++i) {
        REQUIRE(host[cellOffset + i].count == 64);
        REQUIRE(host[cellOffset + i].isFinished == false);
        REQUIRE(host[cellOffset + i].dst != nullptr);
    }
    cellOffset += static_cast<uint64_t >(pow(32, 3));

    for(int i = 0; i < pow(16, 3); ++i) {
        REQUIRE(host[cellOffset + i].count == 512);
        REQUIRE(host[cellOffset + i].isFinished == false);
        REQUIRE(host[cellOffset + i].dst != nullptr);
    }
    cellOffset += static_cast<uint64_t >(pow(16, 3));

    for(int i = 0; i < pow(8, 3); ++i) {
        REQUIRE(host[cellOffset + i].count == 4096);
        REQUIRE(host[cellOffset + i].isFinished == true);
        REQUIRE(host[cellOffset + i].dst == nullptr);
    }
    cellOffset += static_cast<uint64_t >(pow(8, 3));

    for(int i = 0; i < pow(4, 3); ++i) {
        REQUIRE(host[cellOffset + i].count == 0);
        REQUIRE(host[cellOffset + i].isFinished == true);
        REQUIRE(host[cellOffset + i].dst == nullptr);
    }
    cellOffset += static_cast<uint64_t >(pow(4, 3));

    for(int i = 0; i < pow(2, 3); ++i) {
        REQUIRE(host[cellOffset + i].count == 0);
        REQUIRE(host[cellOffset + i].isFinished == true);
        REQUIRE(host[cellOffset + i].dst == nullptr);
    }
    cellOffset += static_cast<uint64_t >(pow(2, 3));

    for(int i = 0; i < pow(1, 3); ++i) {
        REQUIRE(host[cellOffset + i].count == 0);
        REQUIRE(host[cellOffset + i].isFinished == true);
        REQUIRE(host[cellOffset + i].dst == nullptr);
    }
}
