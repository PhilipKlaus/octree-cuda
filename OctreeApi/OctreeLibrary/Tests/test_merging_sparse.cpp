//
// Created by KlausP on 29.10.2020.
//

#include "../src/include/tools.cuh"
#include "catch2/catch.hpp"
#include "octree_processor.cuh"


TEST_CASE ("Test cell merging sparse", "[merging sparse]")
{
    // Create test data point octree
    PointCloudMetadata metadata{};
    unique_ptr<CudaArray<uint8_t>> cloud = tools::generate_point_cloud_cuboid<float> (128, metadata);
    metadata.cloudType                   = CLOUD_FLOAT_UINT8_T;
    metadata.memoryType                  = ClOUD_DEVICE;

    SubsampleMetadata subsampleMetadata{128, true, true};

    // Create the octree
    auto octree = make_unique<OctreeProcessor> (cloud->devicePointer (), 128, 10000, metadata, subsampleMetadata);

    octree->initialPointCounting ();
    octree->performCellMerging ();

    auto denseCount       = octree->getDensePointCountPerVoxel ();
    auto denseToSparseLUT = octree->getDenseToSparseLUT ();
    auto sparseToDenseLUT = octree->getSparseToDenseLUT ();
    auto octreeSparse     = octree->getOctreeSparse ();

    // Require that all cells are filled and that there is no empty space
    REQUIRE (octree->getMetadata ().nodeAmountSparse == 2396745);

    // Require that the point count in the root cell is the sum of all points
    REQUIRE (denseCount[2396744] == 128 * 128 * 128);

    // Check if point counts in each level of detail are correct
    uint32_t cellOffset = 0;

    for (int i = 0; i < pow (128, 3); ++i)
    {
        REQUIRE (denseCount[i] == 1);
        REQUIRE (denseToSparseLUT[i] != -1);
        REQUIRE (octreeSparse[denseToSparseLUT[i]].pointCount == 1);
        REQUIRE (octreeSparse[denseToSparseLUT[i]].isFinished == false);
        REQUIRE (octreeSparse[denseToSparseLUT[i]].parentChunkIndex != 0);
        REQUIRE (octreeSparse[denseToSparseLUT[i]].chunkDataIndex < octree->getMetadata ().cloudMetadata.pointAmount);
        REQUIRE (octreeSparse[denseToSparseLUT[i]].isInternal == false);
        REQUIRE (i == sparseToDenseLUT[denseToSparseLUT[i]]);
    }
    cellOffset += static_cast<uint32_t> (pow (128, 3));

    for (int i = 0; i < pow (64, 3); ++i)
    {
        REQUIRE (denseCount[cellOffset + i] == 8);
        REQUIRE (denseToSparseLUT[cellOffset + i] != -1);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].pointCount == 8);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].isFinished == false);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].parentChunkIndex != 0);
        REQUIRE (
                octreeSparse[denseToSparseLUT[cellOffset + i]].chunkDataIndex <
                octree->getMetadata ().cloudMetadata.pointAmount);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].isInternal == false);
        REQUIRE ((cellOffset + i) == sparseToDenseLUT[denseToSparseLUT[cellOffset + i]]);
    }
    cellOffset += static_cast<uint32_t> (pow (64, 3));

    for (int i = 0; i < pow (32, 3); ++i)
    {
        REQUIRE (denseCount[cellOffset + i] == 64);
        REQUIRE (denseToSparseLUT[cellOffset + i] != -1);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].pointCount == 64);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].isFinished == false);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].parentChunkIndex != 0);
        REQUIRE (
                octreeSparse[denseToSparseLUT[cellOffset + i]].chunkDataIndex <
                octree->getMetadata ().cloudMetadata.pointAmount);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].isInternal == false);
        REQUIRE ((cellOffset + i) == sparseToDenseLUT[denseToSparseLUT[cellOffset + i]]);
    }
    cellOffset += static_cast<uint32_t> (pow (32, 3));

    for (int i = 0; i < pow (16, 3); ++i)
    {
        REQUIRE (denseCount[cellOffset + i] == 512);
        REQUIRE (denseToSparseLUT[cellOffset + i] != -1);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].pointCount == 512);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].isFinished == false);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].parentChunkIndex != 0);
        REQUIRE (
                octreeSparse[denseToSparseLUT[cellOffset + i]].chunkDataIndex <
                octree->getMetadata ().cloudMetadata.pointAmount);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].isInternal == false);
        REQUIRE ((cellOffset + i) == sparseToDenseLUT[denseToSparseLUT[cellOffset + i]]);
    }
    cellOffset += static_cast<uint32_t> (pow (16, 3));

    for (int i = 0; i < pow (8, 3); ++i)
    {
        REQUIRE (denseCount[cellOffset + i] == 4096);
        REQUIRE (denseToSparseLUT[cellOffset + i] != -1);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].pointCount == 4096);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].isFinished == true);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].parentChunkIndex != 0);
        REQUIRE (
                octreeSparse[denseToSparseLUT[cellOffset + i]].chunkDataIndex <
                octree->getMetadata ().cloudMetadata.pointAmount);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].isInternal == false);
        REQUIRE ((cellOffset + i) == sparseToDenseLUT[denseToSparseLUT[cellOffset + i]]);
    }
    cellOffset += static_cast<uint32_t> (pow (8, 3));

    for (int i = 0; i < pow (4, 3); ++i)
    {
        REQUIRE (denseCount[cellOffset + i] == 32768);
        REQUIRE (denseToSparseLUT[cellOffset + i] != -1);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].pointCount == 0);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].isFinished == true);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].parentChunkIndex != 0);
        REQUIRE (
                octreeSparse[denseToSparseLUT[cellOffset + i]].chunkDataIndex <
                octree->getMetadata ().cloudMetadata.pointAmount);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].isInternal == true);
        REQUIRE ((cellOffset + i) == sparseToDenseLUT[denseToSparseLUT[cellOffset + i]]);
    }
    cellOffset += static_cast<uint32_t> (pow (4, 3));

    for (int i = 0; i < pow (2, 3); ++i)
    {
        REQUIRE (denseCount[cellOffset + i] == 262144);
        REQUIRE (denseToSparseLUT[cellOffset + i] != -1);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].pointCount == 0);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].isFinished == true);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].parentChunkIndex != 0);
        REQUIRE (
                octreeSparse[denseToSparseLUT[cellOffset + i]].chunkDataIndex <
                octree->getMetadata ().cloudMetadata.pointAmount);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].isInternal == true);
        REQUIRE ((cellOffset + i) == sparseToDenseLUT[denseToSparseLUT[cellOffset + i]]);
    }
    cellOffset += static_cast<uint32_t> (pow (2, 3));

    for (int i = 0; i < pow (1, 3); ++i)
    {
        REQUIRE (denseCount[cellOffset + i] == 2097152);
        REQUIRE (denseToSparseLUT[cellOffset + i] != -1);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].pointCount == 0);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].isFinished == true);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].parentChunkIndex == 0);
        REQUIRE (
                octreeSparse[denseToSparseLUT[cellOffset + i]].chunkDataIndex <
                octree->getMetadata ().cloudMetadata.pointAmount);
        REQUIRE (octreeSparse[denseToSparseLUT[cellOffset + i]].isInternal == true);
        REQUIRE ((cellOffset + i) == sparseToDenseLUT[denseToSparseLUT[cellOffset + i]]);
        REQUIRE (cellOffset + i == denseToSparseLUT[cellOffset + i]);
        REQUIRE (denseToSparseLUT[cellOffset + i] == 2396744);
    }
}