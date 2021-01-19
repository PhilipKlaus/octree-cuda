//
// Created by KlausP on 13.11.2020.
//

#include <sparseOctree.h>


template <typename coordinateType, typename colorType>
uint32_t SparseOctree<coordinateType, colorType>::exportTreeNode(
        uint8_t *cpuPointCloud,
        const unique_ptr<Chunk[]> &octreeSparse,
        const unique_ptr<uint32_t[]> &dataLUT,
        const string& level,
        uint32_t index,
        const string &folder
) {

    PointCloudMetadata metadata = itsMetadata.cloudMetadata;
    uint32_t count = octreeSparse[index].isParent ? itsSubsampleLUTs[index]->pointCount() : octreeSparse[index].pointCount;
    uint32_t validPoints = count;

    std::unique_ptr<uint32_t[]> lut;
    std::unique_ptr<Averaging[]> averaging;

    if(octreeSparse[index].isParent) {
        lut = itsSubsampleLUTs[index]->toHost();
        averaging = itsAveragingData[index]->toHost();
    }

    for (uint32_t u = 0; u < count; ++u)
    {
        if(octreeSparse[index].isParent) {
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
        std::ofstream ply;
        ply.open (folder + "//" + level + ".ply", std::ios::binary);

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
                if(lut[u] != INVALID_INDEX) {
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[lut[u] * metadata.pointDataStride])), sizeof (float));
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[lut[u] * metadata.pointDataStride + 4])), sizeof (float));
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[lut[u] * metadata.pointDataStride + 8])), sizeof (float));
                    uint8_t r = static_cast<uint8_t>(averaging[u].r / averaging[u].pointCount);
                    ply.write (reinterpret_cast<const char*>(&r), sizeof (uint8_t));
                    uint8_t g = static_cast<uint8_t>(averaging[u].g / averaging[u].pointCount);
                    ply.write (reinterpret_cast<const char*>(&g), sizeof (uint8_t));
                    uint8_t b = static_cast<uint8_t>(averaging[u].b / averaging[u].pointCount);
                    ply.write (reinterpret_cast<const char*>(&b), sizeof (uint8_t));
                    /*ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[lut[u] * metadata.pointDataStride + 12])), sizeof (uint8_t));
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[lut[u] * metadata.pointDataStride + 13])), sizeof (uint8_t));
                    ply.write (reinterpret_cast<const char*> (&(cpuPointCloud[lut[u] * metadata.pointDataStride + 14])), sizeof (uint8_t));*/
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
        if(octreeSparse[index].isFinished) {
            ++itsMetadata.absorbedNodes;
        }
    }
    for(uint32_t i = 0; i < 8; ++i) {
        int childIndex = octreeSparse[index].childrenChunks[i];
        if(childIndex != -1) {
            validPoints += exportTreeNode(cpuPointCloud, octreeSparse, dataLUT, level + std::to_string(i), childIndex, folder);
        }
    }
    return validPoints;
}


template <typename coordinateType, typename colorType>
void SparseOctree<coordinateType, colorType>::exportPlyNodes(const string &folderPath) {
    auto cpuPointCloud = itsCloudData->toHost();
    auto octreeSparse = itsOctree->toHost();
    auto dataLUT = itsDataLUT->toHost();

    uint32_t exportedPoints = exportTreeNode(cpuPointCloud.get(), octreeSparse, dataLUT, string("r"), getRootIndex(), folderPath);
    assert(exportedPoints == itsMetadata.cloudMetadata.pointAmount);
    spdlog::info("Sparse octree ({}/{} points) exported to: {}", exportedPoints, itsMetadata.cloudMetadata.pointAmount, folderPath);
}


template uint32_t SparseOctree<float, uint8_t>::exportTreeNode(
        uint8_t *cpuPointCloud,
        const unique_ptr<Chunk[]> &octreeSparse,
        const unique_ptr<uint32_t[]> &dataLUT,
        const string& level,
        uint32_t index,
        const string &folder
);

template void SparseOctree<float, uint8_t>::exportPlyNodes(const string &folderPath);