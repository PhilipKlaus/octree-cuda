//
// Created by KlausP on 27.10.2020.
//

//
// Created by KlausP on 27.10.2020.
//

#include <denseOctree.h>
#include "catch2/catch.hpp"
#include "../src/tools.cuh"


uint32_t testOctreenode(Vector3 *cpuPointCloud, const unique_ptr<Chunk[]> &octree, const unique_ptr<uint32_t[]> &dataLUT, uint32_t level, uint32_t index) {
    uint32_t count = 0;

    uint32_t previousChunks = 0;
    for(uint32_t i = 7; i > (7 - level); --i) {
        previousChunks += static_cast<uint32_t>(pow(pow(2, i), 3));
    }

    auto newGridSize = static_cast<uint32_t>(pow(2, 7-level));
    auto xy = newGridSize * newGridSize;
    auto z = (index-previousChunks) / xy;
    auto y = (index-previousChunks - (z * xy)) / newGridSize;
    auto x = (index-previousChunks - (z * xy)) % newGridSize;

    auto voxelSize = 128.f / newGridSize;

    if(octree[index].isFinished && octree[index].pointCount > 0) {

        count = octree[index].pointCount;
        for (uint32_t u = 0; u < octree[index].pointCount; ++u)
        {
            Vector3 point = cpuPointCloud[dataLUT[octree[index].chunkDataIndex + u]];
            REQUIRE(point.x > (x * voxelSize));
            REQUIRE(point.x < ((x + 1) * voxelSize));
            REQUIRE(point.y > (y * voxelSize));
            REQUIRE(point.y < ((y + 1) * voxelSize));
            REQUIRE(point.z > (z * voxelSize));
            REQUIRE(point.z < ((z + 1) * voxelSize));
        }
    }
    else {
        if (level > 0) {
            for(uint32_t childrenChunk : octree[index].childrenChunks) {
                count += testOctreenode(cpuPointCloud, octree, dataLUT, level - 1, childrenChunk);
            }
        }
    }
    return count;
}

TEST_CASE ("Test point distributing", "distributing") {

    // Create test data point cloud
    unique_ptr<CudaArray<Vector3>> cuboid = tools::generate_point_cloud_cuboid(128);
    auto cpuData = cuboid->toHost();

    PointCloudMetadata metadata{};
    metadata.pointAmount = 128 * 128 * 128;
    metadata.boundingBox.minimum = Vector3 {0.5, 0.5, 0.5};
    metadata.boundingBox.maximum = Vector3 {127.5, 127.5, 127.5};
    metadata.cloudOffset = Vector3 {0.5, 0.5, 0.5};
    metadata.scale = {1.f, 1.f, 1.f};

    auto cloud = make_unique<DenseOctree>(metadata, move(cuboid));

    cloud->initialPointCounting(7);
    cloud->performCellMerging(10000); // All points reside in the 4th level (8x8x8) of the octree
    cloud->distributePoints();

    auto octree = cloud->getOctreeDense();
    auto dataLUT = cloud->getDataLUT();
    uint32_t topLevelIndex = 2396745-1;

    REQUIRE(testOctreenode(cpuData.get(), octree, dataLUT, 7, topLevelIndex) == cloud->getMetadata().pointAmount);
}
