//
// Created by KlausP on 28.10.2020.
//

#include <sparseOctree.h>
#include "catch2/catch.hpp"
#include "tools.cuh"

TEST_CASE ("Test initial sparse point counting", "[counting sparse]") {

    // Create test data point octree
    PointCloudMetadata metadata{};
    unique_ptr<CudaArray<uint8_t>> cloud = tools::generate_point_cloud_cuboid(128, metadata);

    // Create the octree
    auto octree = make_unique<SparseOctree>(OctreeTypes::GRID_128, OctreeTypes::GRID_128, 10000, metadata, move(cloud));

    // Perform initial point counting
    octree->initialPointCounting();

    auto denseCount = octree->getDensePointCountPerVoxel();
    auto denseToSparseLUT = octree->getDenseToSparseLUT();

    // Require that all cells are filled and that there is no empty space
    REQUIRE(octree->getMetadata().nodeAmountSparse == pow(128, 3));

    // Require that the sum of the accumulated point counts equaly to the actual point amount of the octree
    uint32_t sum = 0;
    for(uint32_t i = 0; i < static_cast<uint32_t>(pow(128, 3)); ++i) {
        sum += denseCount[i];
        // We can assume that there is 1 point within each node
        REQUIRE(denseCount[i] == 1);
        // We can assume that there exist a sparse index for each dense index as there are no empty cells
        REQUIRE(denseToSparseLUT[i] != -1);
    }
    REQUIRE(sum == octree->getMetadata().cloudMetadata.pointAmount);
}
