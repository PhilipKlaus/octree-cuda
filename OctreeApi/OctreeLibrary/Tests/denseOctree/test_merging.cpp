#include <denseOctree.h>
#include "catch2/catch.hpp"
#include "tools.cuh"
#include "../../src/defines.cuh"


TEST_CASE ("Test point merging ", "[merging dense]") {

    // Create test data point cloud
    unique_ptr<CudaArray<Vector3>> cuboid = tools::generate_point_cloud_cuboid(128);

    PointCloudMetadata metadata{};
    metadata.pointAmount = 128 * 128 * 128;
    metadata.boundingBox.minimum = Vector3 {0.5, 0.5, 0.5};
    metadata.boundingBox.maximum = Vector3 {127.5, 127.5, 127.5};
    metadata.cloudOffset = Vector3 {0.5, 0.5, 0.5};
    metadata.scale = {1.f, 1.f, 1.f};

    auto cloud = make_unique<DenseOctree>(metadata, move(cuboid));

    cloud->initialPointCounting(7);
    cloud->performCellMerging(10000);

    // Test if each point fall exactly in one cell
    auto octree = cloud->getOctreeDense();
    uint32_t cellOffset = 0;

    for(int i = 0; i < pow(128, 3); ++i) {
        REQUIRE(octree[i].pointCount == 1);
        REQUIRE(octree[i].isFinished == false);
        REQUIRE(octree[i].parentChunkIndex != INVALID_INDEX);
    }
    cellOffset += static_cast<uint32_t >(pow(128, 3));

    for(int i = 0; i < pow(64, 3); ++i) {
        REQUIRE(octree[cellOffset + i].pointCount == 8);
        REQUIRE(octree[cellOffset + i].isFinished == false);
        REQUIRE(octree[cellOffset + i].parentChunkIndex != INVALID_INDEX);
    }
    cellOffset += static_cast<uint32_t >(pow(64, 3));

    for(int i = 0; i < pow(32, 3); ++i) {
        REQUIRE(octree[cellOffset + i].pointCount == 64);
        REQUIRE(octree[cellOffset + i].isFinished == false);
        REQUIRE(octree[cellOffset + i].parentChunkIndex != INVALID_INDEX);
    }
    cellOffset += static_cast<uint32_t >(pow(32, 3));

    for(int i = 0; i < pow(16, 3); ++i) {
        REQUIRE(octree[cellOffset + i].pointCount == 512);
        REQUIRE(octree[cellOffset + i].isFinished == false);
        REQUIRE(octree[cellOffset + i].parentChunkIndex != INVALID_INDEX);
    }
    cellOffset += static_cast<uint32_t >(pow(16, 3));

    for(int i = 0; i < pow(8, 3); ++i) {
        REQUIRE(octree[cellOffset + i].pointCount == 4096);
        REQUIRE(octree[cellOffset + i].isFinished == true);
        REQUIRE(octree[cellOffset + i].parentChunkIndex != INVALID_INDEX);
    }
    cellOffset += static_cast<uint32_t >(pow(8, 3));

    for(int i = 0; i < pow(4, 3); ++i) {
        REQUIRE(octree[cellOffset + i].pointCount == 0);
        REQUIRE(octree[cellOffset + i].isFinished == true);
        REQUIRE(octree[cellOffset + i].parentChunkIndex != INVALID_INDEX);
    }
    cellOffset += static_cast<uint32_t >(pow(4, 3));

    for(int i = 0; i < pow(2, 3); ++i) {
        REQUIRE(octree[cellOffset + i].pointCount == 0);
        REQUIRE(octree[cellOffset + i].isFinished == true);
        REQUIRE(octree[cellOffset + i].parentChunkIndex != INVALID_INDEX);
    }
    cellOffset += static_cast<uint32_t >(pow(2, 3));

    for(int i = 0; i < pow(1, 3); ++i) {
        REQUIRE(octree[cellOffset + i].pointCount == 0);
        REQUIRE(octree[cellOffset + i].isFinished == true);
        REQUIRE(octree[cellOffset + i].parentChunkIndex == 0);
    }
}
