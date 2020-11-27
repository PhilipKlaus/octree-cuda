//
// Created by KlausP on 13.11.2020.
//

#include <sparseOctree.h>

uint32_t SparseOctree::exportTreeNode(
        uint8_t *cpuPointCloud,
        const unique_ptr<Chunk[]> &octreeSparse,
        const unique_ptr<uint32_t[]> &dataLUT,
        uint32_t level,
        uint32_t index,
        const string &folder
) {

    PointCloudMetadata metadata = itsMetadata.cloudMetadata;
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
               "property uchar red\n"
               "property uchar green\n"
               "property uchar blue\n"
               "end_header\n";
        for (uint32_t u = 0; u < count; ++u)
        {
            if(octreeSparse[index].isParent) {
                auto lut = itsSubsampleLUTs[index]->toHost();
                if(lut[u] != INVALID_INDEX) {
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[lut[u] * metadata.pointDataStride])), sizeof (float));
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[lut[u] * metadata.pointDataStride + 4])), sizeof (float));
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[lut[u] * metadata.pointDataStride + 8])), sizeof (float));
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[lut[u] * metadata.pointDataStride + 12])), sizeof (uint8_t));
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[lut[u] * metadata.pointDataStride + 13])), sizeof (uint8_t));
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[lut[u] * metadata.pointDataStride + 14])), sizeof (uint8_t));
                }
            }
            else {
                if(dataLUT[octreeSparse[index].chunkDataIndex + u] != INVALID_INDEX) {
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[dataLUT[octreeSparse[index].chunkDataIndex + u] * metadata.pointDataStride])), sizeof (float));
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[dataLUT[octreeSparse[index].chunkDataIndex + u] * metadata.pointDataStride + 4])), sizeof (float));
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[dataLUT[octreeSparse[index].chunkDataIndex + u] * metadata.pointDataStride + 8])), sizeof (float));
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[dataLUT[octreeSparse[index].chunkDataIndex + u] * metadata.pointDataStride + 12])), sizeof (uint8_t));
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[dataLUT[octreeSparse[index].chunkDataIndex + u] * metadata.pointDataStride + 13])), sizeof (uint8_t));
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[dataLUT[octreeSparse[index].chunkDataIndex + u] * metadata.pointDataStride + 14])), sizeof (uint8_t));
                }
            }
        }
        ply.close ();
    }
    else {
        validPoints = 0;
    }
    if (level > 0) {
        for(uint32_t i = 0; i < octreeSparse[index].childrenChunksCount; ++i) {
            validPoints += exportTreeNode(cpuPointCloud, octreeSparse, dataLUT, level - 1, octreeSparse[index].childrenChunks[i], folder);
        }
    }
    return validPoints;
}

void SparseOctree::exportOctree(const string &folderPath) {
    auto cpuPointCloud = itsCloudData->toHost();
    auto octreeSparse = itsOctreeSparse->toHost();
    auto dataLUT = itsDataLUT->toHost();

    // ToDo: Remove .get() -> pass unique_ptr by reference
    uint32_t exportedPoints = exportTreeNode(cpuPointCloud.get(), octreeSparse, dataLUT, itsMetadata.depth, getRootIndex(), folderPath);
    assert(exportedPoints == itsPointCloudMetadata.pointAmount);
    spdlog::info("Sparse octree ({}/{} points) exported to: {}", exportedPoints, itsMetadata.cloudMetadata.pointAmount, folderPath);
    spdlog::info("{}", itsSubsampleLUTs.size());
}