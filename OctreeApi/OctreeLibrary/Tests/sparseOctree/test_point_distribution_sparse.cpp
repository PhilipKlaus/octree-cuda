//
// Created by KlausP on 04.11.2020.
//

#include <sparseOctree.h>
#include "catch2/catch.hpp"
#include "tools.cuh"


uint32_t testOctreenodeSparse(uint8_t *cpuPointCloud, const unique_ptr<Chunk[]> &octree, const unique_ptr<uint32_t[]> &dataLUT, const unique_ptr<int[]> &sparseToDense, uint32_t level, uint32_t index) {
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
            auto *point = reinterpret_cast<Vector3*>(cpuPointCloud + dataLUT[octree[index].chunkDataIndex + u] * 12);
            REQUIRE(point->x > (coord.x * voxelSize));
            REQUIRE(point->x < ((coord.x + 1) * voxelSize));
            REQUIRE(point->y > (coord.y * voxelSize));
            REQUIRE(point->y < ((coord.y + 1) * voxelSize));
            REQUIRE(point->z > (coord.z * voxelSize));
            REQUIRE(point->z < ((coord.z + 1) * voxelSize));
        }
    }
    else {
        if (level > 0) {
            for(uint32_t i = 0; i < octree[index].childrenChunksCount; ++i) {
                count += testOctreenodeSparse(cpuPointCloud, octree, dataLUT, sparseToDense, level - 1, octree[index].childrenChunks[i]);
            }
        }
    }
    return count;
}

TEST_CASE ("Test point distributing sparse", "[distributing sparse]") {

    // Create test data point cloud
    PointCloudMetadata metadata{};
    unique_ptr<CudaArray<uint8_t>> cuboid = tools::generate_point_cloud_cuboid(128, metadata);
    auto cpuData = cuboid->toHost();

    auto cloud = make_unique<SparseOctree>(7, 10000, metadata, move(cuboid));

    cloud->initialPointCounting();
    cloud->performCellMerging(); // All points reside in the 4th level (8x8x8) of the octree
    cloud->distributePoints();

    auto octree = cloud->getOctreeSparse();
    auto dataLUT = cloud->getDataLUT();
    auto sparseToDenseLUT = cloud->getSparseToDenseLUT();
    uint32_t topLevelIndex = cloud->getMetadata().nodeAmountSparse-1;

    REQUIRE(testOctreenodeSparse(cpuData.get(), octree, dataLUT, sparseToDenseLUT, 7, topLevelIndex) == cloud->getMetadata().cloudMetadata.pointAmount);
}
