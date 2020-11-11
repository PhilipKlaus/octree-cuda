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

    uint32_t count = octreeSparse[index].isParent ? itsSubsampleLUTs[index]->pointCount() : octreeSparse[index].pointCount;
    uint32_t validPoints = count;
    for (uint32_t u = 0; u < count; ++u)
    {
        if(octreeSparse[index].isParent) {
            auto lut = itsSubsampleLUTs[index]->toHost();
            if(lut[u] == INVALID_INDEX) {
                --validPoints;
            }
        }
        else {
            if(dataLUT[octreeSparse[index].chunkDataIndex + u] == INVALID_INDEX) {
                --validPoints;
            }
        }
    }

    if(octreeSparse[index].isFinished && validPoints > 0) {
        string name = octreeSparse[index].isParent ? (folder + "//_parent_") : (folder + "//_leaf_");
        std::ofstream ply;
        ply.open (name + std::to_string(level) +
                  "_" +
                  std::to_string(index) +
                  "_" +
                  std::to_string(validPoints) +
                  ".ply",
                  std::ios::binary
        );

        ply << "ply\n"
               "format binary_little_endian 1.0\n"
               "comment Created by AIT Austrian Institute of Technology\n"
               "element vertex "
            << validPoints
            << "\n"
               "property float x\n"
               "property float y\n"
               "property float z\n"
               "end_header\n";
        for (uint32_t u = 0; u < count; ++u)
        {
            if(octreeSparse[index].isParent) {
                auto lut = itsSubsampleLUTs[index]->toHost();
                if(lut[u] != INVALID_INDEX) {
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[lut[u]].x)), sizeof (float));
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[lut[u]].y)), sizeof (float));
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[lut[u]].z)), sizeof (float));
                }
            }
            else {
                if(dataLUT[octreeSparse[index].chunkDataIndex + u] != INVALID_INDEX) {
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[dataLUT[octreeSparse[index].chunkDataIndex + u]].x)), sizeof (float));
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[dataLUT[octreeSparse[index].chunkDataIndex + u]].y)), sizeof (float));
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[dataLUT[octreeSparse[index].chunkDataIndex + u]].z)), sizeof (float));
                }
            }
        }
        ply.close ();
    }
    if (level > 0) {
        for(int i = 0; i < octreeSparse[index].childrenChunksCount; ++i) {
            count += exportTreeNode(cpuPointCloud, octreeSparse, dataLUT, level - 1, octreeSparse[index].childrenChunks[i], folder);
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
    spdlog::info("{}", itsSubsampleLUTs.size());
}

void SparseOctree::freeGpuMemory() {
    itsDensePointCountPerVoxel.reset();
    itsDenseToSparseLUT.reset();
    itsSparseToDenseLUT.reset();
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

unique_ptr<int[]> SparseOctree::getSparseToDenseLUT(){
    return itsSparseToDenseLUT->toHost();
}

unique_ptr<Chunk[]> SparseOctree::getOctreeSparse() {
    return itsOctreeSparse->toHost();
}

uint32_t SparseOctree::getVoxelAmountSparse() {
    return itsVoxelAmountSparse->toHost()[0];
}

unordered_map<uint32_t, unique_ptr<CudaArray<uint32_t>>> const& SparseOctree::getSubsampleLUT() const {
    return itsSubsampleLUTs;
}
