#include "catch2/catch.hpp"
#include "octree.cuh"

void fillDummyOctree (Node (&chunks)[9])
{
    /*
     *                   50
     *       /   /   /  /   \  \  \   \
     *      100 350 0 800 1300 0 700 250
     */

    chunks[8].childNodes[0] = 0;
    chunks[8].childNodes[1] = 1;
    chunks[8].childNodes[2] = -1;
    chunks[8].childNodes[3] = 3;
    chunks[8].childNodes[4] = 4;
    chunks[8].childNodes[5] = -1;
    chunks[8].childNodes[6] = 6;
    chunks[8].childNodes[7] = 7;
    chunks[8].pointCount    = 50;
    chunks[8].isFinished    = true;
    chunks[8].isParent      = true;

    chunks[0].pointCount = 100;
    chunks[1].pointCount = 350;
    chunks[2].pointCount = 0;
    chunks[3].pointCount = 800;
    chunks[4].pointCount = 1300;
    chunks[5].pointCount = 0;
    chunks[6].pointCount = 700;
    chunks[7].pointCount = 250;

    for (auto i = 0; i < 8; ++i)
    {
        chunks[i].isParent   = false;
        chunks[i].isFinished = true;
        for (int& childrenChunk : chunks[i].childNodes)
        {
            childrenChunk = -1;
        }
    }
}

TEST_CASE ("An Octree should")
{
    OctreeData octree (512);

    SECTION ("before creating the hierarchy")
    {
        SECTION ("return correct Node Statistics (not updated)")
        {
            auto& meta = octree.getNodeStatistics ();
            CHECK (meta.depth == 9);
            CHECK (meta.nodeAmountSparse == 0);
            CHECK (meta.nodeAmountDense == 153391689);
            CHECK (meta.parentNodeAmount == 0);
            CHECK (meta.minPointsPerNode == 0);
            CHECK (meta.maxPointsPerNode == 0);
            CHECK (meta.leafNodeAmount == 0);
            CHECK (meta.parentNodeAmount == 0);
            CHECK (meta.meanPointsPerLeafNode == 0);
            CHECK (meta.maxLeafDepth == 0);
        }

        SECTION ("throw a HierarchyNotCreatedException when calling getRootIndex")
        {
            CHECK_THROWS_AS (octree.getRootIndex (), HierarchyNotCreatedException);
        }

        SECTION ("throw a HierarchyNotCreatedException when calling getNode")
        {
            CHECK_THROWS_AS (octree.getNode (0), HierarchyNotCreatedException);
        }

        SECTION ("throw a HierarchyNotCreatedException when calling copyToHost")
        {
            CHECK_THROWS_AS (octree.copyToHost (), HierarchyNotCreatedException);
        }

        SECTION ("throw a HierarchyNotCreatedException when calling getHost")
        {
            CHECK_THROWS_AS (octree.getHost (), HierarchyNotCreatedException);
        }

        SECTION ("throw a HierarchyNotCreatedException when calling getDevice")
        {
            CHECK_THROWS_AS (octree.getDevice (), HierarchyNotCreatedException);
        }

        SECTION ("throw a updateNodeStatistics when calling updateNodeStatistics")
        {
            CHECK_THROWS_AS (octree.updateNodeStatistics (), HierarchyNotCreatedException);
        }

        SECTION ("return correct not amounts")
        {
            CHECK (octree.getNodeAmount (0) == std::pow (512, 3));
            CHECK (octree.getNodeAmount (1) == std::pow (256, 3));
            CHECK (octree.getNodeAmount (2) == std::pow (128, 3));
            CHECK (octree.getNodeAmount (3) == std::pow (64, 3));
            CHECK (octree.getNodeAmount (4) == std::pow (32, 3));
            CHECK (octree.getNodeAmount (5) == std::pow (16, 3));
            CHECK (octree.getNodeAmount (6) == std::pow (8, 3));
            CHECK (octree.getNodeAmount (7) == std::pow (4, 3));
            CHECK (octree.getNodeAmount (8) == std::pow (2, 3));
            CHECK (octree.getNodeAmount (9) == std::pow (1, 3));
        }

        SECTION ("return correct grid sizes")
        {
            CHECK (octree.getGridSize (0) == 512);
            CHECK (octree.getGridSize (1) == 256);
            CHECK (octree.getGridSize (2) == 128);
            CHECK (octree.getGridSize (3) == 64);
            CHECK (octree.getGridSize (4) == 32);
            CHECK (octree.getGridSize (5) == 16);
            CHECK (octree.getGridSize (6) == 8);
            CHECK (octree.getGridSize (7) == 4);
            CHECK (octree.getGridSize (8) == 2);
            CHECK (octree.getGridSize (9) == 1);
        }

        SECTION ("return correct node offsets")
        {
            CHECK (octree.getNodeOffset (0) == 0);
            CHECK (octree.getNodeOffset (1) == 134217728);
            CHECK (octree.getNodeOffset (2) == 150994944);
            CHECK (octree.getNodeOffset (3) == 153092096);
            CHECK (octree.getNodeOffset (4) == 153354240);
            CHECK (octree.getNodeOffset (5) == 153387008);
            CHECK (octree.getNodeOffset (6) == 153391104);
            CHECK (octree.getNodeOffset (7) == 153391616);
            CHECK (octree.getNodeOffset (8) == 153391680);
            CHECK (octree.getNodeOffset (9) == 153391688);
        }
    }
    SECTION ("after creating the hierarchy")
    {
        octree.createHierarchy (9);

        SECTION ("return correct Node Statistics (not updated)")
        {
            auto& meta = octree.getNodeStatistics ();
            CHECK (meta.nodeAmountSparse == 9);
            CHECK (meta.nodeAmountDense == 153391689);
            CHECK (meta.parentNodeAmount == 0);
            CHECK (meta.minPointsPerNode == 0);
            CHECK (meta.maxPointsPerNode == 0);
            CHECK (meta.leafNodeAmount == 0);
            CHECK (meta.parentNodeAmount == 0);
            CHECK (meta.meanPointsPerLeafNode == 0);
            CHECK (meta.maxLeafDepth == 0);
        }

        SECTION ("return correct Node Statistics (updated)")
        {
            Node chunks[9];
            fillDummyOctree (chunks);

            REQUIRE (
                    cudaMemcpy (
                            octree.getDevice (),
                            reinterpret_cast<uint8_t*> (chunks),
                            sizeof (Node) * 9,
                            cudaMemcpyHostToDevice) == cudaSuccess);

            octree.updateNodeStatistics ();
            auto& meta = octree.getNodeStatistics ();
            CHECK (meta.nodeAmountSparse == 9);
            CHECK (meta.nodeAmountDense == 153391689);
            CHECK (meta.parentNodeAmount == 1);
            CHECK (meta.minPointsPerNode == 100);
            CHECK (meta.maxPointsPerNode == 1300);
            CHECK (meta.leafNodeAmount == 6);
            CHECK (meta.parentNodeAmount == 1);
            CHECK (meta.meanPointsPerLeafNode == Approx (583.3).epsilon (0.001));
            CHECK (meta.maxLeafDepth == 1);
        }

        SECTION ("return correct root index")
        {
            CHECK (octree.getRootIndex () == 8);
        }

        SECTION ("copy the octree to host")
        {
            // Copy a dummy octree to gpu
            Node chunks[9];
            fillDummyOctree (chunks);

            REQUIRE (
                    cudaMemcpy (
                            octree.getDevice (),
                            reinterpret_cast<uint8_t*> (chunks),
                            sizeof (Node) * 9,
                            cudaMemcpyHostToDevice) == cudaSuccess);

            auto host = octree.getHost ();
            CHECK (host[octree.getRootIndex ()].childNodes[0] == 0);
            CHECK (host[octree.getRootIndex ()].childNodes[1] == 1);
            CHECK (host[octree.getRootIndex ()].childNodes[2] == -1);
            CHECK (host[octree.getRootIndex ()].childNodes[3] == 3);
            CHECK (host[octree.getRootIndex ()].childNodes[4] == 4);
            CHECK (host[octree.getRootIndex ()].childNodes[5] == -1);
            CHECK (host[octree.getRootIndex ()].childNodes[6] == 6);
            CHECK (host[octree.getRootIndex ()].childNodes[7] == 7);
            CHECK (host[octree.getRootIndex ()].isParent == true);
            CHECK (host[octree.getRootIndex ()].pointCount == 50);
            CHECK (octree.getNode (0).pointCount == 100);
            CHECK (octree.getNode (1).pointCount == 350);
            CHECK (octree.getNode (2).pointCount == 0);
            CHECK (octree.getNode (3).pointCount == 800);
            CHECK (octree.getNode (4).pointCount == 1300);
            CHECK (octree.getNode (5).pointCount == 0);
            CHECK (octree.getNode (6).pointCount == 700);
            CHECK (octree.getNode (7).pointCount == 250);
        }
    }
}