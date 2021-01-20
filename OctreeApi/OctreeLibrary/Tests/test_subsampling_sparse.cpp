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
        for(uint32_t i = 0; i < 8; ++i) {
            if(chunk.childrenChunks[i] != -1) {
                testSubsampleTree(octree, subsampleLUT, chunk.childrenChunks[i], level - 1);
            }
        }
    }
}

TEST_CASE ("Test node subsampling", "[subsampling]") {

    // Create test data point cloud
    PointCloudMetadata metadata{};
    unique_ptr<CudaArray<uint8_t>> cuboid = tools::generate_point_cloud_cuboid(128, metadata);
    auto cpuData = cuboid->toHost();

    // Create the octree
    auto octree = make_unique<SparseOctree<float, uint8_t>>(GRID_128, GRID_128, 10000, metadata, RANDOM_POINT);
    octree->setPointCloudDevice(move(cuboid));

    octree->initialPointCounting();
    octree->performCellMerging(); // All points reside in the 3th level (8x8x8) of the octree
    octree->distributePoints();
    octree->performSubsampling();

   // Ensure that for each relevant parent node exists a evaluateSubsamples data Lut
   REQUIRE(octree->getSubsampleLUT().size() == pow(4, 3) + pow(2, 3) + pow(1, 3));

    auto octreeData = octree->getOctreeSparse();
    uint32_t topLevelIndex = octree->getMetadata().nodeAmountSparse - 1;
    testSubsampleTree(octreeData, octree->getSubsampleLUT(), topLevelIndex, 7);
}
