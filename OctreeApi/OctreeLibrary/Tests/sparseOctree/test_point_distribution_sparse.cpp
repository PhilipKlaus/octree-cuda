//
// Created by KlausP on 04.11.2020.
//

#include <sparseOctree.h>
#include "catch2/catch.hpp"
#include "tools.cuh"


uint32_t testOctreenodeSparse(Vector3 *cpuPointCloud, const unique_ptr<Chunk[]> &octree, const unique_ptr<uint32_t[]> &dataLUT, const unique_ptr<int[]> &sparseToDense, uint32_t level, uint32_t index) {
    uint32_t count = 0;

    uint32_t previousChunks = 0;
    for(uint32_t i = 7; i > (7 - level); --i) {
        previousChunks += static_cast<uint32_t>(pow(pow(2, i), 3));
    }

    int denseIndex = sparseToDense[index];

    auto newGridSize = static_cast<uint32_t>(pow(2, 7-level));

    Vector3i coord{};
    auto indexInVoxel = denseIndex - previousChunks;
    tools::mapFromDenseIdxToDenseCoordinates(coord, indexInVoxel, newGridSize);

    auto voxelSize = 128.f / newGridSize;

    if(octree[index].isFinished && octree[index].pointCount > 0) {

        count = octree[index].pointCount;
        for (uint32_t u = 0; u < octree[index].pointCount; ++u)
        {
            Vector3 point = cpuPointCloud[dataLUT[octree[index].chunkDataIndex + u]];
            REQUIRE(point.x > (coord.x * voxelSize));
            REQUIRE(point.x < ((coord.x + 1) * voxelSize));
            REQUIRE(point.y > (coord.y * voxelSize));
            REQUIRE(point.y < ((coord.y + 1) * voxelSize));
            REQUIRE(point.z > (coord.z * voxelSize));
            REQUIRE(point.z < ((coord.z + 1) * voxelSize));
        }
    }
    else {
        if (level > 0) {
            for(int i = 0; i < octree[index].childrenChunksCount; ++i) {
                count += testOctreenodeSparse(cpuPointCloud, octree, dataLUT, sparseToDense, level - 1, octree[index].childrenChunks[i]);
            }
        }
    }
    return count;
}

TEST_CASE ("Test point distributing sparse", "[distributing sparse]") {

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
    cloud->performCellMerging(10000); // All points reside in the 4th level (8x8x8) of the octree
    cloud->distributePoints();

    auto octree = cloud->getOctreeSparse();
    auto dataLUT = cloud->getDataLUT();
    auto sparseToDenseLUT = cloud->getSparseToDenseLUT();
    uint32_t topLevelIndex = cloud->getVoxelAmountSparse()-1;

    REQUIRE(testOctreenodeSparse(cpuData.get(), octree, dataLUT, sparseToDenseLUT, 7, topLevelIndex) == cloud->getMetadata().pointAmount);
}
