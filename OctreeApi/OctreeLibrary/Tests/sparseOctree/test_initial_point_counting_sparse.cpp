//
// Created by KlausP on 28.10.2020.
//

#include <sparseOctree.h>
#include "catch2/catch.hpp"
#include "tools.cuh"

TEST_CASE ("Test initial sparse point counting", "[counting sparse]") {

    // Create test data point cloud
    unique_ptr<CudaArray<Vector3>> cuboid = tools::generate_point_cloud_cuboid(256);

    PointCloudMetadata metadata{};
    metadata.pointAmount = 256 * 256 * 256;
    metadata.boundingBox.minimum = Vector3 {0.5, 0.5, 0.5};
    metadata.boundingBox.maximum = Vector3 {255.5, 255.5, 255.5};
    metadata.cloudOffset = Vector3 {0.5, 0.5, 0.5};
    metadata.scale = {1.f, 1.f, 1.f};

    auto cloud = make_unique<SparseOctree>(7, 10000, metadata, move(cuboid));

    cloud->initialPointCounting();

    auto denseCount = cloud->getDensePointCountPerVoxel();
    auto denseToSparseLUT = cloud->getDenseToSparseLUT();

    // Require that all cells are filled and that there is no empty space
    REQUIRE(cloud->getMetadata().nodeAmountSparse == pow(128, 3));

    // Require that the sum of the accumulated point counts equaly to the actual point amount of the cloud
    uint32_t sum = 0;
    for(int i = 0; i < pow(128, 3); ++i) {
        REQUIRE(denseCount[i] == 8);
        sum += denseCount[i];
        // We can assume that there exist a sparse index for each dense index as there are no empty cells
        REQUIRE(denseToSparseLUT[i] != -1);
    }
    REQUIRE(sum == cloud->getMetadata().cloudMetadata.pointAmount);
}
