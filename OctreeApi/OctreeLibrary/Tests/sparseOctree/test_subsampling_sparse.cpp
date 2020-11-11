//
// Created by KlausP on 06.11.2020.
//

#include <sparseOctree.h>
#include "catch2/catch.hpp"

void testSubsampleTree(
        const unique_ptr<Chunk[]> &octree,
        unordered_map<uint32_t,
        unique_ptr<CudaArray<uint32_t>>> const& subsampleLUT,
        uint32_t index,
        uint32_t level) {

    Chunk chunk = octree[index];

    if(chunk.isParent) {
        switch(level) {
            case 7:
                REQUIRE(subsampleLUT.at(index)->pointCount() == 128 * 128 * 128);
                break;
            case 6:
                REQUIRE(subsampleLUT.at(index)->pointCount() == 64 * 64 * 64);
                break;
            case 5:
                REQUIRE(subsampleLUT.at(index)->pointCount() == 32 * 32 * 32);
                break;
            default:
                break;
        }
        for(uint32_t i = 0; i < chunk.childrenChunksCount; ++i) {
            testSubsampleTree(octree, subsampleLUT, chunk.childrenChunks[i], level - 1);
        }
    }
}

TEST_CASE ("Test node subsampling", "[subsampling]") {

    // Create test data point cloud
    unique_ptr<CudaArray<Vector3>> cuboid = tools::generate_point_cloud_cuboid(128);
    auto cpuData = cuboid->toHost();

    PointCloudMetadata metadata{};
    metadata.pointAmount = 128 * 128 * 128;
    metadata.boundingBox.minimum = Vector3 {0.5, 0.5, 0.5};
    metadata.boundingBox.maximum = Vector3 {127.5, 127.5, 127.5};
    metadata.cloudOffset = Vector3 {0.5, 0.5, 0.5};
    metadata.scale = {1.f, 1.f, 1.f};

    auto cloud = make_unique<SparseOctree>(metadata, move(cuboid));

    cloud->initialPointCounting(7);
    cloud->performCellMerging(10000); // All points reside in the 3th level (8x8x8) of the octree
    cloud->distributePoints();
    cloud->performIndexing();

   // Ensure that for each relevant parent node exists a subsample data Lut
   REQUIRE(cloud->getSubsampleLUT().size() == pow(4,3) + pow(2,3) + pow(1,3));

    auto octree = cloud->getOctreeSparse();
    uint32_t topLevelIndex = cloud->getVoxelAmountSparse()-1;
    testSubsampleTree(octree, cloud->getSubsampleLUT(), topLevelIndex, 7);
}
