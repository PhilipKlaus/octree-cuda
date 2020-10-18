#include "catch2/catch.hpp"
#include "../lib/src/tools.cuh"
#include "pointcloud.h"


TEST_CASE ("Test initial point counting and merging ", "[counting + merging]") {

    // Create test data point cloud
    unique_ptr<CudaArray<Vector3>> cuboid = generate_point_cloud_cuboid(128);

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
    cloud->performCellMerging(10000);

    // Test if each point fall exactly in one cell
    auto host = cloud->getCountingGrid();
    for(int i = 0; i < pow(128, 3); ++i) {
        REQUIRE(host[0][i].count == 1);
        REQUIRE(host[0][i].isFinished == false);
        REQUIRE(host[0][i].dst != nullptr);
    }
    for(int i = 0; i < pow(64, 3); ++i) {
        REQUIRE(host[1][i].count == 8);
        REQUIRE(host[1][i].isFinished == false);
        REQUIRE(host[1][i].dst != nullptr);
    }
    for(int i = 0; i < pow(32, 3); ++i) {
        REQUIRE(host[2][i].count == 64);
        REQUIRE(host[2][i].isFinished == false);
        REQUIRE(host[2][i].dst != nullptr);
    }
    for(int i = 0; i < pow(16, 3); ++i) {
        REQUIRE(host[3][i].count == 512);
        REQUIRE(host[3][i].isFinished == false);
        REQUIRE(host[3][i].dst != nullptr);
    }
    for(int i = 0; i < pow(8, 3); ++i) {
        REQUIRE(host[4][i].count == 4096);
        REQUIRE(host[4][i].isFinished == true);
        REQUIRE(host[4][i].dst == nullptr);
    }
    for(int i = 0; i < pow(4, 3); ++i) {
        REQUIRE(host[5][i].count == 0);
        REQUIRE(host[5][i].isFinished == true);
        REQUIRE(host[5][i].dst == nullptr);
    }
    for(int i = 0; i < pow(2, 3); ++i) {
        REQUIRE(host[6][i].count == 0);
        REQUIRE(host[6][i].isFinished == true);
        REQUIRE(host[6][i].dst == nullptr);
    }
    for(int i = 0; i < pow(1, 3); ++i) {
        REQUIRE(host[7][i].count == 0);
        REQUIRE(host[7][i].isFinished == true);
        REQUIRE(host[7][i].dst == nullptr);
    }
}
