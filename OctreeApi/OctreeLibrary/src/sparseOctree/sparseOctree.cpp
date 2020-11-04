//
// Created by KlausP on 04.11.2020.
//

#include <sparseOctree.h>

uint32_t SparseOctree::exportTreeNode(
        Vector3 *cpuPointCloud,
        const unique_ptr<Chunk[]> &octreeSparse,
        const unique_ptr<uint32_t[]> &dataLUT,
        uint32_t level,
        uint32_t index,
        const string &folder
        ) {

    uint32_t count = 0;

    if(octreeSparse[index].isFinished && octreeSparse[index].pointCount > 0) {
        count = octreeSparse[index].pointCount;
        std::ofstream ply;
        ply.open (folder +
                "\\tree_" + std::to_string(level) +
                "_" +
                std::to_string(index) +
                "_" +
                std::to_string(octreeSparse[index].pointCount) +
                ".ply",
                std::ios::binary
        );

        ply << "ply\n"
               "format binary_little_endian 1.0\n"
               "comment Created by AIT Austrian Institute of Technology\n"
               "element vertex "
            << octreeSparse[index].pointCount
            << "\n"
               "property float x\n"
               "property float y\n"
               "property float z\n"
               "end_header\n";
        for (uint32_t u = 0; u < octreeSparse[index].pointCount; ++u)
        {
            ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[dataLUT[octreeSparse[index].chunkDataIndex + u]].x)), sizeof (float));
            ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[dataLUT[octreeSparse[index].chunkDataIndex + u]].y)), sizeof (float));
            ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[dataLUT[octreeSparse[index].chunkDataIndex + u]].z)), sizeof (float));
        }
        ply.close ();
    }
    else {
        if (level > 0) {
            for(int i = 0; i < octreeSparse[index].childrenChunksCount; ++i) {
                count += exportTreeNode(cpuPointCloud, octreeSparse, dataLUT, level - 1, octreeSparse[index].childrenChunks[i], folder);
            }
        }
    }
    return count;
}

void SparseOctree::exportOctree(const string &folderPath) {
    auto cpuPointCloud = itsCloudData->toHost();
    auto octreeSparse = itsOctreeSparse->toHost();
    auto dataLUT = itsDataLUT->toHost();
    uint32_t topLevelIndex = itsVoxelAmountSparse->toHost()[0] - 1;

    // ToDo: Remove .get() -> pass unique_ptr by reference
    uint32_t exportedPoints = exportTreeNode(cpuPointCloud.get(), octreeSparse, dataLUT, itsGlobalOctreeDepth, topLevelIndex, folderPath);
    assert(exportedPoints == itsMetadata.pointAmount);
    spdlog::info("Sparse octree ({}/{} points) exported to: {}", exportedPoints, itsMetadata.pointAmount, folderPath);
}

void SparseOctree::freeGpuMemory() {
    itsDensePointCountPerVoxel.reset();
    itsDenseToSparseLUT.reset();
    itsVoxelAmountSparse.reset();
    itsOctreeSparse.reset();
    spdlog::debug("Sparse octree GPU memory deleted");
}

unique_ptr<uint32_t[]> SparseOctree::getDensePointCountPerVoxel() {
    return itsDensePointCountPerVoxel->toHost();
}

unique_ptr<int[]> SparseOctree::getDenseToSparseLUT() {
    return itsDenseToSparseLUT->toHost();
}

unique_ptr<Chunk[]> SparseOctree::getOctreeSparse() {
    return itsOctreeSparse->toHost();
}

uint32_t SparseOctree::getVoxelAmountSparse() {
    return itsVoxelAmountSparse->toHost()[0];
}