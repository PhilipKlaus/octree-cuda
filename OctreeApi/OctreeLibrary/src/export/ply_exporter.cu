#include "ply_exporter.cuh"


template <typename coordinateType, typename colorType>
PlyExporter<coordinateType, colorType>::PlyExporter (
        const GpuArrayU8& pointCloud,
        const GpuOctree& octree,
        const GpuArrayU32& leafeLut,
        const unordered_map<uint32_t, GpuArrayU32>& parentLut,
        const unordered_map<uint32_t, GpuAveraging>& parentAveraging,
        OctreeMetadata metadata) :
        OctreeExporter<coordinateType, colorType> (pointCloud, octree, leafeLut, parentLut, parentAveraging, metadata),
        itsPointsExported (0)
{}

template <typename coordinateType, typename colorType>
void PlyExporter<coordinateType, colorType>::exportOctree (const std::string& path)
{
    exportNode (this->getRootIndex (), "r", path);
    spdlog::info ("Exported {}/{} points to: {}", itsPointsExported, this->itsMetadata.cloudMetadata.pointAmount, path);
}

template <typename coordinateType, typename colorType>
void PlyExporter<coordinateType, colorType>::exportNode (
        uint32_t nodeIndex, const string& octreeLevel, const std::string& path)
{
    bool isParent = this->itsOctree[nodeIndex].isParent;
    bool isFinished = this->itsOctree[nodeIndex].isFinished;
    bool isAveraging = true;

    PointCloudMetadata cloudMetadata = this->itsMetadata.cloudMetadata;
    uint32_t pointsToExport = isParent ? this->itsParentLutCounts[nodeIndex] : this->itsOctree[nodeIndex].pointCount;
    const std::unique_ptr<uint32_t[]>& lut = isParent ? this->itsParentLut[nodeIndex] : this->itsLeafeLut;

    // ToDo: Only neccessary in additive mode
    //pointsToExport      = getValidPointAmount (nodeIndex, pointsToExport);
    uint32_t dataStride = cloudMetadata.pointDataStride;

    uint32_t coordinateSize = sizeof (coordinateType);
    uint32_t colorSize      = sizeof (colorType);

    if (isFinished && pointsToExport > 0)
    {
        std::ofstream ply;
        ply.open (path + R"(/)" + octreeLevel + ".ply", std::ios::binary);

        string header;
        createPlyHeader (header, pointsToExport);

        ply << header;


        for (uint32_t u = 0; u < pointsToExport; ++u)
        {
            if (this->itsOctree[nodeIndex].isParent)
            {
                if (lut[u] != INVALID_INDEX)
                {
                    uint32_t pointIndex = lut[u] * dataStride;
                    uint32_t byteOffset = 0;

                    ply.write (reinterpret_cast<const char*> (&(this->itsPointCloud[pointIndex])), coordinateSize);

                    byteOffset += coordinateSize;
                    ply.write (
                            reinterpret_cast<const char*> (&(this->itsPointCloud[pointIndex + byteOffset])), coordinateSize);

                    byteOffset += coordinateSize;
                    ply.write (
                            reinterpret_cast<const char*> (&(this->itsPointCloud[pointIndex + byteOffset])), coordinateSize);

                    byteOffset += coordinateSize;

                    if (isAveraging)
                    {
                        uint32_t sumPointCount = this->itsAveraging[nodeIndex][u].pointCount;

                        auto r = static_cast<colorType> (this->itsAveraging[nodeIndex][u].r / sumPointCount);
                        ply.write (reinterpret_cast<const char*> (&r), colorSize);
                        auto g = static_cast<colorType> (this->itsAveraging[nodeIndex][u].g / sumPointCount);
                        ply.write (reinterpret_cast<const char*> (&g), colorSize);
                        auto b = static_cast<colorType> (this->itsAveraging[nodeIndex][u].b / sumPointCount);
                        ply.write (reinterpret_cast<const char*> (&b), colorSize);
                    }

                    else
                    {

                        ply.write (
                                reinterpret_cast<const char*> (&(this->itsPointCloud[pointIndex + byteOffset])), colorSize);

                        byteOffset += colorSize;
                        ply.write (
                                reinterpret_cast<const char*> (&(this->itsPointCloud[pointIndex + byteOffset])), colorSize);

                        byteOffset += colorSize;
                        ply.write (
                                reinterpret_cast<const char*> (&(this->itsPointCloud[pointIndex + byteOffset])), colorSize);
                    }
                }
            }
            else
            {
                if (lut[this->itsOctree[nodeIndex].chunkDataIndex + u] != INVALID_INDEX)
                {
                    uint32_t byteOffset = 0;
                    uint32_t pointIndex = lut[this->itsOctree[nodeIndex].chunkDataIndex + u] * dataStride;

                    ply.write (
                            reinterpret_cast<const char*> (&(this->itsPointCloud[pointIndex])), coordinateSize);

                    byteOffset += coordinateSize;
                    ply.write (
                            reinterpret_cast<const char*> (&(this->itsPointCloud[pointIndex + byteOffset])), coordinateSize);

                    byteOffset += coordinateSize;
                    ply.write (
                            reinterpret_cast<const char*> (&(this->itsPointCloud[pointIndex + byteOffset])), coordinateSize);

                    byteOffset += coordinateSize;
                    ply.write (
                            reinterpret_cast<const char*> (&(this->itsPointCloud[pointIndex + byteOffset])), coordinateSize);

                    byteOffset += colorSize;
                    ply.write (
                            reinterpret_cast<const char*> (&(this->itsPointCloud[pointIndex + byteOffset])), coordinateSize);

                    byteOffset += colorSize;
                    ply.write (
                            reinterpret_cast<const char*> (&(this->itsPointCloud[pointIndex + byteOffset])), coordinateSize);
                }
            }
        }
        ply.close ();
    }
    else
    {
        if (this->itsOctree[nodeIndex].isFinished)
        {
            ++this->itsAbsorbedNodes;
        }
    }
    for (uint32_t i = 0; i < 8; ++i)
    {
        int childIndex = this->itsOctree[nodeIndex].childrenChunks[i];
        if (childIndex != -1)
        {
            exportNode(childIndex, octreeLevel + std::to_string (i), path);
        }
    }
    this->itsPointsExported += pointsToExport;
}

template <typename coordinateType, typename colorType>
uint32_t PlyExporter<coordinateType, colorType>::getValidPointAmount (uint32_t nodeIndex, uint32_t pointAmount)
{
    uint32_t validPoints = pointAmount;

    for (uint32_t u = 0; u < pointAmount; ++u)
    {
        if (itsOctree[nodeIndex].isParent)
        {
            if (itsParentLut[nodeIndex][u] == INVALID_INDEX)
            {
                --validPoints;
            }
        }
        else
        {
            if (itsLeafeLut[itsOctree[nodeIndex].chunkDataIndex + u] == INVALID_INDEX)
            {
                --validPoints;
            }
        }
    }
    return validPoints;
}

template <typename coordinateType, typename colorType>
void PlyExporter<coordinateType, colorType>::createPlyHeader (string& header, uint32_t pointsToExport)
{
    string coordinateType;
    string colorType;

    switch (itsMetadata.cloudMetadata.cloudType)
    {
    case CLOUD_FLOAT_UINT8_T:
        coordinateType = "float";
        colorType      = "uchar";
        break;
    default:
        coordinateType = "double";
        colorType      = "uchar";
        break;
    };

    header = "ply\n"
             "format binary_little_endian 1.0\n"
             "comment Created by AIT Austrian Institute of Technology\n"
             "element vertex " +
             to_string (pointsToExport);

    header += "\n"
              "property " +
              coordinateType +
              " x\n"
              "property " +
              coordinateType +
              " y\n"
              "property " +
              coordinateType +
              " z\n"
              "property " +
              colorType +
              " red\n"
              "property " +
              colorType +
              " green\n"
              "property " +
              colorType +
              " blue\n"
              "end_header\n";
}


//----------------------------------------------------------------------------------------------------------------------
//                                           SparseOctree<float, uint8_t>
//----------------------------------------------------------------------------------------------------------------------
template PlyExporter<float, uint8_t>::PlyExporter(
        const GpuArrayU8& pointCloud,
        const GpuOctree& octree,
        const GpuArrayU32& leafeLut,
        const unordered_map<uint32_t, GpuArrayU32>& parentLut,
        const unordered_map<uint32_t, GpuAveraging>& parentAveraging,
        OctreeMetadata metadata);

template void PlyExporter<float, uint8_t>::exportOctree(const std::string& path);

//----------------------------------------------------------------------------------------------------------------------
//                                           SparseOctree<double, uint8_t>
//----------------------------------------------------------------------------------------------------------------------
template PlyExporter<double, uint8_t>::PlyExporter(
        const GpuArrayU8& pointCloud,
        const GpuOctree& octree,
        const GpuArrayU32& leafeLut,
        const unordered_map<uint32_t, GpuArrayU32>& parentLut,
        const unordered_map<uint32_t, GpuAveraging>& parentAveraging,
        OctreeMetadata metadata);

template void PlyExporter<double, uint8_t>::exportOctree(const std::string& path);