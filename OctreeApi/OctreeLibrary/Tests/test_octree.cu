#include "catch2/catch.hpp"
#include "octree.cuh"

TEST_CASE ("An Octree should")
{
    Octree octree(512, 10000);

    SECTION("before creating the hierarchy") {

        SECTION("return correct OctreeMetadata") {
            auto &meta = octree.getMetadata();
            CHECK(meta.mergingThreshold == 10000);
            CHECK(meta.nodeAmountSparse == 0);
            CHECK(meta.nodeAmountDense == 153391689);
            CHECK(meta.chunkingGrid == 512);
            CHECK(meta.depth == 9);
        }

        SECTION("throw a HierarchyNotCreatedException when calling getRootIndex") {
            CHECK_THROWS_AS(octree.getRootIndex(), HierarchyNotCreatedException);
        }

        SECTION("throw a HierarchyNotCreatedException when calling getNode") {
            CHECK_THROWS_AS(octree.getNode(0), HierarchyNotCreatedException);
        }

        SECTION("throw a HierarchyNotCreatedException when calling copyToHost") {
            CHECK_THROWS_AS(octree.copyToHost(), HierarchyNotCreatedException);
        }

        SECTION("throw a HierarchyNotCreatedException when calling getHost") {
            CHECK_THROWS_AS(octree.getHost(), HierarchyNotCreatedException);
        }

        SECTION("throw a HierarchyNotCreatedException when calling getDevice") {
            CHECK_THROWS_AS(octree.getDevice(), HierarchyNotCreatedException);
        }

        SECTION("throw a updateNodeStatistics when calling updateNodeStatistics") {
            CHECK_THROWS_AS(octree.updateNodeStatistics(), HierarchyNotCreatedException);
        }

        SECTION("return correct not amounts") {
            CHECK(octree.getNodeAmount(0) == std::pow(512, 3));
            CHECK(octree.getNodeAmount(1) == std::pow(256, 3));
            CHECK(octree.getNodeAmount(2) == std::pow(128, 3));
            CHECK(octree.getNodeAmount(3) == std::pow(64, 3));
            CHECK(octree.getNodeAmount(4) == std::pow(32, 3));
            CHECK(octree.getNodeAmount(5) == std::pow(16, 3));
            CHECK(octree.getNodeAmount(6) == std::pow(8, 3));
            CHECK(octree.getNodeAmount(7) == std::pow(4, 3));
            CHECK(octree.getNodeAmount(8) == std::pow(2, 3));
            CHECK(octree.getNodeAmount(9) == std::pow(1, 3));
        }

        SECTION("return correct grid sizes") {
            CHECK(octree.getGridSize(0) == 512);
            CHECK(octree.getGridSize(1) == 256);
            CHECK(octree.getGridSize(2) == 128);
            CHECK(octree.getGridSize(3) == 64);
            CHECK(octree.getGridSize(4) == 32);
            CHECK(octree.getGridSize(5) == 16);
            CHECK(octree.getGridSize(6) == 8);
            CHECK(octree.getGridSize(7) == 4);
            CHECK(octree.getGridSize(8) == 2);
            CHECK(octree.getGridSize(9) == 1);
        }

        SECTION("return correct node offsets") {
            CHECK(octree.getNodeOffset(0) == 0);
            CHECK(octree.getNodeOffset(1) == 134217728);
            CHECK(octree.getNodeOffset(2) == 150994944);
            CHECK(octree.getNodeOffset(3) == 153092096);
            CHECK(octree.getNodeOffset(4) == 153354240);
            CHECK(octree.getNodeOffset(5) == 153387008);
            CHECK(octree.getNodeOffset(6) == 153391104);
            CHECK(octree.getNodeOffset(7) == 153391616);
            CHECK(octree.getNodeOffset(8) == 153391680);
            CHECK(octree.getNodeOffset(9) == 153391688);
        }

        SECTION("return uninitialized Node Statistics") {
            auto &meta = octree.getNodeStatistics();
            CHECK(meta.parentNodeAmount == 0);
            CHECK(meta.minPointsPerNode == 0);
            CHECK(meta.maxPointsPerNode == 0);
            CHECK(meta.leafNodeAmount == 0);
            CHECK(meta.parentNodeAmount == 0);
            CHECK(meta.meanPointsPerLeafNode == 0);
        }
    }
    SECTION("after creating the hierarchy") {
        octree.createHierarchy(9);

        SECTION("return correct OctreeMetadata") {
            auto &meta = octree.getMetadata();
            CHECK(meta.mergingThreshold == 10000);
            CHECK(meta.nodeAmountSparse == 9);
            CHECK(meta.nodeAmountDense == 153391689);
            CHECK(meta.chunkingGrid == 512);
            CHECK(meta.depth == 9);
        }

        SECTION("return correct root index") {
            CHECK(octree.getRootIndex() == 8);
        }

        SECTION("copy the octree to host") {

            // Copy a dummy octree to gpu
            Chunk chunks[9];
            chunks[8].childrenChunks[0] = 1;
            chunks[8].isParent = true;
            chunks[8].pointCount = 100;
            chunks[1].isParent = true;
            chunks[1].pointCount = 300;

            REQUIRE (cudaMemcpy (octree.getDevice(), reinterpret_cast<uint8_t*>(chunks), sizeof (Chunk) * 9, cudaMemcpyHostToDevice) == cudaSuccess);

            auto host = octree.getHost();
            REQUIRE (host[octree.getRootIndex()].childrenChunks[0] == 1);
            REQUIRE (host[octree.getRootIndex()].isParent == true);
            REQUIRE (host[octree.getRootIndex()].pointCount == 100);

            REQUIRE (octree.getNode(1).isParent == true);
            REQUIRE (octree.getNode(1).pointCount == 300);
        }
    }
}