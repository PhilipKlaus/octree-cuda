#include "ply_exporter.cuh"


template <typename coordinateType, typename colorType>
PlyExporter<coordinateType, colorType>::PlyExporter (
        const GpuArrayU8& pointCloud,
        const GpuOctree& octree,
        const GpuArrayU32& leafeLut,
        const unordered_map<uint32_t, GpuArrayU32>& parentLut,
        const unordered_map<uint32_t, GpuAveraging>& parentAveraging,
        OctreeMetadata<coordinateType> metadata) :
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
    bool isParent    = this->itsOctree[nodeIndex].isParent;
    bool isFinished  = this->itsOctree[nodeIndex].isFinished;

    // ToDo: read from config + change in kernel;
    bool isAveraging = true;
    bool isReplacement = true;

    PointCloudMetadata<coordinateType> cloudMetadata = this->itsMetadata.cloudMetadata;
    uint32_t pointsToExport = isParent ? this->itsParentLutCounts[nodeIndex] : this->itsOctree[nodeIndex].pointCount;
    const std::unique_ptr<uint32_t[]>& lut = isParent ? this->itsParentLut[nodeIndex] : this->itsLeafeLut;

    if(!isReplacement) {
        pointsToExport      = getValidPointAmount (nodeIndex, pointsToExport);
    }

    uint32_t dataStride = cloudMetadata.pointDataStride;

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
                    uint32_t pointByteIndex = lut[u] * dataStride;
                    writePointCoordinates (ply, pointByteIndex);

                    if (isAveraging)
                    {
                        const std::unique_ptr<Averaging[]>& averaging = this->itsAveraging[nodeIndex];
                        writeColorAveraged (ply, nodeIndex, u);
                    }

                    else
                    {
                        writeColorNonAveraged(ply, pointByteIndex);
                    }
                }
            }
            else
            {
                if (lut[this->itsOctree[nodeIndex].chunkDataIndex + u] != INVALID_INDEX)
                {
                    uint32_t pointByteIndex = lut[this->itsOctree[nodeIndex].chunkDataIndex + u] * dataStride;
                    writePointCoordinates(ply, pointByteIndex);
                    writeColorNonAveraged(ply, pointByteIndex);
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
            exportNode (childIndex, octreeLevel + std::to_string (i), path);
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

template <typename coordinateType, typename colorType>
void PlyExporter<coordinateType, colorType>::writePointCoordinates (std::ofstream& file, uint32_t pointByteIndex)
{
    uint32_t coordinateSize = sizeof (coordinateType);

    file.write (reinterpret_cast<const char*> (&(this->itsPointCloud[pointByteIndex])), coordinateSize);

    pointByteIndex += coordinateSize;
    file.write (reinterpret_cast<const char*> (&(this->itsPointCloud[pointByteIndex])), coordinateSize);

    pointByteIndex += coordinateSize;
    file.write (reinterpret_cast<const char*> (&(this->itsPointCloud[pointByteIndex])), coordinateSize);
}

template <typename coordinateType, typename colorType>
void PlyExporter<coordinateType, colorType>::writeColorAveraged (
        std::ofstream& file, uint32_t nodeIndex, uint32_t pointIndex)
{
    uint32_t sumPointCount = this->itsAveraging[nodeIndex][pointIndex].pointCount;

    auto r = static_cast<colorType> (this->itsAveraging[nodeIndex][pointIndex].r / sumPointCount);
    file.write (reinterpret_cast<const char*> (&r), sizeof (colorType));
    auto g = static_cast<colorType> (this->itsAveraging[nodeIndex][pointIndex].g / sumPointCount);
    file.write (reinterpret_cast<const char*> (&g), sizeof (colorType));
    auto b = static_cast<colorType> (this->itsAveraging[nodeIndex][pointIndex].b / sumPointCount);
    file.write (reinterpret_cast<const char*> (&b), sizeof (colorType));
}

template <typename coordinateType, typename colorType>
void PlyExporter<coordinateType, colorType>::writeColorNonAveraged (std::ofstream& file, uint32_t pointByteIndex)
{
    pointByteIndex += sizeof (coordinateType) * 3;
    uint32_t colorSize = sizeof (colorType);

    file.write (reinterpret_cast<const char*> (&(this->itsPointCloud[pointByteIndex])), colorSize);

    pointByteIndex += colorSize;
    file.write (reinterpret_cast<const char*> (&(this->itsPointCloud[pointByteIndex])), colorSize);

    pointByteIndex += colorSize;
    file.write (reinterpret_cast<const char*> (&(this->itsPointCloud[pointByteIndex])), colorSize);
}

//----------------------------------------------------------------------------------------------------------------------
//                                           SparseOctree<float, uint8_t>
//----------------------------------------------------------------------------------------------------------------------
template PlyExporter<float, uint8_t>::PlyExporter (
        const GpuArrayU8& pointCloud,
        const GpuOctree& octree,
        const GpuArrayU32& leafeLut,
        const unordered_map<uint32_t, GpuArrayU32>& parentLut,
        const unordered_map<uint32_t, GpuAveraging>& parentAveraging,
        OctreeMetadata<float> metadata);

template void PlyExporter<float, uint8_t>::exportOctree (const std::string& path);

//----------------------------------------------------------------------------------------------------------------------
//                                           SparseOctree<double, uint8_t>
//----------------------------------------------------------------------------------------------------------------------
template PlyExporter<double, uint8_t>::PlyExporter (
        const GpuArrayU8& pointCloud,
        const GpuOctree& octree,
        const GpuArrayU32& leafeLut,
        const unordered_map<uint32_t, GpuArrayU32>& parentLut,
        const unordered_map<uint32_t, GpuAveraging>& parentAveraging,
        OctreeMetadata<double> metadata);

template void PlyExporter<double, uint8_t>::exportOctree (const std::string& path);