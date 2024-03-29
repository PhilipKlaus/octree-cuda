/*#include "ply_exporter.cuh"


template <typename coordinateType, typename colorType>
PlyExporter<coordinateType, colorType>::PlyExporter (
        const PointCloud& pointCloud,
        const std::shared_ptr<Chunk[]>& octree,
        OctreeMetadata metadata,
        PointCloudMetadata cloudMetadata,
        SubsampleMetadata subsamplingMetadata) :
        OctreeExporter<coordinateType, colorType> (
                pointCloud, octree, metadata, cloudMetadata, subsamplingMetadata)
{}

template <typename coordinateType, typename colorType>
void PlyExporter<coordinateType, colorType>::exportOctree (const std::string& path)
{
    exportNode (this->getRootIndex (), "r", path);
    spdlog::info ("Exported {}/{} points to: {}", itsPointsExported, this->itsCloudMetadata.pointAmount, path);
}

template <typename coordinateType, typename colorType>
void PlyExporter<coordinateType, colorType>::exportNode (
        uint32_t nodeIndex, const string& octreeLevel, const std::string& path)
{
    bool isInternal   = this->itsOctree[nodeIndex].isInternal;
    bool isFinished = this->itsOctree[nodeIndex].isFinished;

    // ToDo: read from config + change in kernel;
    bool isAveraging = true;

    uint32_t pointsInNode                  = this->getPointsInNode (nodeIndex);
    const std::unique_ptr<uint32_t[]>& lut = this->itsSubsamples->getLutHost (
            nodeIndex); // isInternal ? this->itsSubsamples->getLutHost (nodeIndex) : this->itsLeafLut;

    uint32_t dataStride = this->itsCloudMetadata.pointDataStride;

    if (isFinished)
    {
        auto buffer = std::make_unique<uint8_t[]> (pointsInNode * (3 * (sizeof (coordinateType) + sizeof (colorType))));
        uint32_t validPoints  = 0;
        uint64_t bufferOffset = 0;

        for (uint32_t u = 0; u < pointsInNode; ++u)
        {
            //  Export parent node
            if (this->itsOctree[nodeIndex].isInternal)
            {
                if (lut[u] != INVALID_INDEX)
                {
                    ++validPoints;
                    uint64_t pointByteIndex = lut[u] * dataStride;
                    writePointCoordinates (buffer, bufferOffset, pointByteIndex);
                    bufferOffset += (3 * sizeof (coordinateType));

                    if (isAveraging)
                    {
                        const std::unique_ptr<uint64_t[]>& averaging = this->itsSubsamples->getAvgHost (nodeIndex);
                        writeColorAveraged (buffer, bufferOffset, nodeIndex, u);
                    }

                    else
                    {
                        writeColorNonAveraged (buffer, bufferOffset, pointByteIndex);
                    }
                    bufferOffset += (3 * sizeof (colorType));
                }
            }

            //  Export child node
            else
            {
                if (lut[this->itsOctree[nodeIndex].chunkDataIndex + u] != INVALID_INDEX)
                {
                    ++validPoints;

                    uint32_t pointByteIndex = lut[this->itsOctree[nodeIndex].chunkDataIndex + u] * dataStride;
                    writePointCoordinates (buffer, bufferOffset, pointByteIndex);
                    bufferOffset += (3 * sizeof (coordinateType));
                    writeColorNonAveraged (buffer, bufferOffset, pointByteIndex);
                    bufferOffset += (3 * sizeof (colorType));
                }
            }
        }
        this->itsPointsExported += validPoints;
        std::ofstream ply;
        ply.open (path + R"(/)" + octreeLevel + ".ply", std::ios::binary);
        string header;
        createPlyHeader (header, validPoints);
        ply << header;
        ply.write (
                reinterpret_cast<const char*> (&buffer[0]),
                validPoints * (3 * (sizeof (coordinateType) + sizeof (colorType))));
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
}


template <typename coordinateType, typename colorType>
void PlyExporter<coordinateType, colorType>::createPlyHeader (string& header, uint32_t pointsToExport)
{
    string coordinateType;
    string colorType;

    switch (this->itsCloudMetadata.cloudType)
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
void PlyExporter<coordinateType, colorType>::writePointCoordinates (
        const std::unique_ptr<uint8_t[]>& buffer, uint64_t bufferOffset, uint64_t pointByteIndex)
{
    uint32_t coordinateSize = sizeof (coordinateType);

    std::memcpy (buffer.get () + bufferOffset, &(this->itsCloud[pointByteIndex]), coordinateSize);

    bufferOffset += coordinateSize;
    pointByteIndex += coordinateSize;
    std::memcpy (buffer.get () + bufferOffset, &(this->itsCloud[pointByteIndex]), coordinateSize);

    bufferOffset += coordinateSize;
    pointByteIndex += coordinateSize;
    std::memcpy (buffer.get () + bufferOffset, &(this->itsCloud[pointByteIndex]), coordinateSize);
}

template <typename coordinateType, typename colorType>
void PlyExporter<coordinateType, colorType>::writeColorAveraged (
        const std::unique_ptr<uint8_t[]>& buffer, uint64_t bufferOffset, uint32_t nodeIndex, uint32_t pointIndex)
{
    uint8_t colorSize = sizeof (colorType);
    uint64_t encoded  = this->itsSubsamples->getAvgHost (nodeIndex)[pointIndex];

    auto r = static_cast<colorType> (encoded >> 46);
    std::memcpy (buffer.get () + bufferOffset, &r, colorSize);

    bufferOffset += colorSize;
    auto g = static_cast<colorType> (encoded >> 28);
    std::memcpy (buffer.get () + bufferOffset, &g, colorSize);

    bufferOffset += colorSize;
    auto b = static_cast<colorType> (encoded >> 10);
    std::memcpy (buffer.get () + bufferOffset, &b, colorSize);
}

template <typename coordinateType, typename colorType>
void PlyExporter<coordinateType, colorType>::writeColorNonAveraged (
        const std::unique_ptr<uint8_t[]>& buffer, uint64_t bufferOffset, uint64_t pointByteIndex)
{
    pointByteIndex += sizeof (coordinateType) * 3;
    uint32_t colorSize = sizeof (colorType);

    std::memcpy (buffer.get () + bufferOffset, &(this->itsCloud[pointByteIndex]), colorSize);

    bufferOffset += colorSize;
    pointByteIndex += colorSize;
    std::memcpy (buffer.get () + bufferOffset, &(this->itsCloud[pointByteIndex]), colorSize);

    bufferOffset += colorSize;
    pointByteIndex += colorSize;
    std::memcpy (buffer.get () + bufferOffset, &(this->itsCloud[pointByteIndex]), colorSize);
}

//----------------------------------------------------------------------------------------------------------------------
//                                           SparseOctree<float, uint8_t>
//----------------------------------------------------------------------------------------------------------------------
template PlyExporter<float, uint8_t>::PlyExporter (
        const PointCloud& pointCloud,
        const std::shared_ptr<Chunk[]>& octree,
        OctreeMetadata metadata,
        PointCloudMetadata cloudMetadata,
        SubsampleMetadata subsamplingMetadata);

template void PlyExporter<float, uint8_t>::exportOctree (const std::string& path);

//----------------------------------------------------------------------------------------------------------------------
//                                           SparseOctree<double, uint8_t>
//----------------------------------------------------------------------------------------------------------------------
template PlyExporter<double, uint8_t>::PlyExporter (
        const PointCloud& pointCloud,
        const std::shared_ptr<Chunk[]>& octree,
        OctreeMetadata metadata,
        PointCloudMetadata cloudMetadata,
        SubsampleMetadata subsamplingMetadata);

template void PlyExporter<double, uint8_t>::exportOctree (const std::string& path);
 */