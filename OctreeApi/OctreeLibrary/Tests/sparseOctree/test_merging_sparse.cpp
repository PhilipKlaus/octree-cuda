//
// Created by KlausP on 29.10.2020.
//

#include "catch2/catch.hpp"
#include "tools.cuh"
#include "sparseOctree.h"


TEST_CASE ("Test cell merging sparse", "[merging sparse]") {

    // Create test data point cloud
    unique_ptr<CudaArray<Vector3>> cuboid = tools::generate_point_cloud_cuboid(256);

    PointCloudMetadata metadata{};
    metadata.pointAmount = 256 * 256 * 256;
    metadata.boundingBox.minimum = Vector3 {0.5, 0.5, 0.5};
    metadata.boundingBox.maximum = Vector3 {255.5, 255.5, 255.5};
    metadata.cloudOffset = Vector3 {0.5, 0.5, 0.5};
    metadata.scale = {1.f, 1.f, 1.f};

    auto cloud = make_unique<SparseOctree>(metadata, move(cuboid));

    cloud->initialPointCounting(7);
    cloud->performCellMerging(40000);

    auto denseCount = cloud->getDensePointCountPerVoxel();
    auto denseToSparseLUT = cloud->getDenseToSparseLUT();
    auto sparseToDenseLUT = cloud->getSparseToDenseLUT();
    auto octreeSparse = cloud->getOctreeSparse();

    // Require that all cells are filled and that there is no empty space
    REQUIRE(cloud->getVoxelAmountSparse() == 2396745);

    // Require that the point count in the root cell is the sum of all points
    REQUIRE(denseCount[2396744] == 256 * 256 * 256);

    // Check if point counts in each level of detail are correct
    uint32_t cellOffset = 0;

    for(int i = 0; i < pow(128, 3); ++i) {
        REQUIRE(denseCount[i] == 8);
        REQUIRE(denseToSparseLUT[i] != -1);
        REQUIRE(octreeSparse[denseToSparseLUT[i]].pointCount == 8);
        REQUIRE(octreeSparse[denseToSparseLUT[i]].isFinished == false);
        REQUIRE(octreeSparse[denseToSparseLUT[i]].childrenChunksCount == 0);
        REQUIRE(octreeSparse[denseToSparseLUT[i]].parentChunkIndex != 0);
        REQUIRE(octreeSparse[denseToSparseLUT[i]].chunkDataIndex < cloud->getMetadata().pointAmount);
        REQUIRE(octreeSparse[denseToSparseLUT[i]].isParent == false);
        REQUIRE(i == sparseToDenseLUT[denseToSparseLUT[i]]);
    }
    cellOffset += static_cast<uint32_t >(pow(128, 3));

    for(int i = 0; i < pow(64, 3); ++i) {
        REQUIRE(denseCount[cellOffset + i] == 64);
        REQUIRE(denseToSparseLUT[cellOffset + i] != -1);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].pointCount == 64);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].isFinished == false);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].childrenChunksCount == 8);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].parentChunkIndex != 0);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].chunkDataIndex < cloud->getMetadata().pointAmount);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].isParent == false);
        REQUIRE((cellOffset + i) == sparseToDenseLUT[denseToSparseLUT[cellOffset + i]]);
    }
    cellOffset += static_cast<uint32_t >(pow(64, 3));

    for(int i = 0; i < pow(32, 3); ++i) {
        REQUIRE(denseCount[cellOffset + i] == 512);
        REQUIRE(denseToSparseLUT[cellOffset + i] != -1);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].pointCount == 512);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].isFinished == false);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].childrenChunksCount == 8);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].parentChunkIndex != 0);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].chunkDataIndex < cloud->getMetadata().pointAmount);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].isParent == false);
        REQUIRE((cellOffset + i) == sparseToDenseLUT[denseToSparseLUT[cellOffset + i]]);
    }
    cellOffset += static_cast<uint32_t >(pow(32, 3));

    for(int i = 0; i < pow(16, 3); ++i) {
        REQUIRE(denseCount[cellOffset + i] == 4096);
        REQUIRE(denseToSparseLUT[cellOffset + i] != -1);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].pointCount == 4096);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].isFinished == false);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].childrenChunksCount == 8);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].parentChunkIndex != 0);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].chunkDataIndex < cloud->getMetadata().pointAmount);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].isParent == false);
        REQUIRE((cellOffset + i) == sparseToDenseLUT[denseToSparseLUT[cellOffset + i]]);
    }
    cellOffset += static_cast<uint32_t >(pow(16, 3));

    for(int i = 0; i < pow(8, 3); ++i) {
        REQUIRE(denseCount[cellOffset + i] == 32768);
        REQUIRE(denseToSparseLUT[cellOffset + i] != -1);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].pointCount == 32768);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].isFinished == true);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].childrenChunksCount == 8);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].parentChunkIndex != 0);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].chunkDataIndex < cloud->getMetadata().pointAmount);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].isParent == false);
        REQUIRE((cellOffset + i) == sparseToDenseLUT[denseToSparseLUT[cellOffset + i]]);
    }
    cellOffset += static_cast<uint32_t >(pow(8, 3));

    for(int i = 0; i < pow(4, 3); ++i) {
        REQUIRE(denseCount[cellOffset + i] == 262144);
        REQUIRE(denseToSparseLUT[cellOffset + i] != -1);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].pointCount == 0);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].isFinished == true);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].childrenChunksCount == 8);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].parentChunkIndex != 0);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].chunkDataIndex < cloud->getMetadata().pointAmount);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].isParent == true);
        REQUIRE((cellOffset + i) == sparseToDenseLUT[denseToSparseLUT[cellOffset + i]]);
    }
    cellOffset += static_cast<uint32_t >(pow(4, 3));

    for(int i = 0; i < pow(2, 3); ++i) {
        REQUIRE(denseCount[cellOffset + i] == 2097152);
        REQUIRE(denseToSparseLUT[cellOffset + i] != -1);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].pointCount == 0);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].isFinished == true);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].childrenChunksCount == 8);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].parentChunkIndex != 0);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].chunkDataIndex < cloud->getMetadata().pointAmount);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].isParent == true);
        REQUIRE((cellOffset + i) == sparseToDenseLUT[denseToSparseLUT[cellOffset + i]]);
    }
    cellOffset += static_cast<uint32_t >(pow(2, 3));

    for(int i = 0; i < pow(1, 3); ++i) {
        REQUIRE(denseCount[cellOffset + i] == 16777216);
        REQUIRE(denseToSparseLUT[cellOffset + i] != -1);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].pointCount == 0);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].isFinished == true);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].childrenChunksCount == 8);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].parentChunkIndex == 0);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].chunkDataIndex < cloud->getMetadata().pointAmount);
        REQUIRE(octreeSparse[denseToSparseLUT[cellOffset + i]].isParent == true);
        REQUIRE((cellOffset + i) == sparseToDenseLUT[denseToSparseLUT[cellOffset + i]]);
        REQUIRE(cellOffset + i == denseToSparseLUT[cellOffset + i]);
        REQUIRE(denseToSparseLUT[cellOffset + i] == 2396744);
    }
}