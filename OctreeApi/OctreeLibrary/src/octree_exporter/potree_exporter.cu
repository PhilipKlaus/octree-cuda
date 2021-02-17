#include "potree_exporter.cuh"
#include "thread_pool.h"
#include <iomanip>
#include <iostream>
#include <json.hpp>
#include <list>
#include <queue>
#include <unordered_map>

constexpr char METADATA_FILE_NAME[]    = "/metadata.json";
constexpr char HIERARCHY_FILE_NAME[]   = "/hierarchy.bin";
constexpr char POINT_FILE_NAME[]       = "/octree.bin";
constexpr char POTREE_DATA_VERSION[]   = "2.0";
constexpr char POTREE_DATA_ENCODING[]  = "default";
constexpr uint8_t HIERARCHY_NODE_BYTES = 22;
constexpr uint8_t HIERARCHY_STEP_SIZE  = 100;
constexpr uint8_t HIERARCHY_DEPTH      = 20;

// POSITIONS
constexpr char POSITION_NAME[]          = "position";
constexpr char POSITION_TYPE[]          = "int32";
constexpr uint8_t POSITION_ELEMENTS     = 3;
constexpr uint8_t POSITION_ELEMENT_SIZE = sizeof (uint32_t);
constexpr uint8_t POSITION_SIZE         = POSITION_ELEMENT_SIZE * 3;

// COLORS
constexpr char COLOR_NAME[]          = "rgb";
constexpr char COLOR_TYPE[]          = "uint16";
constexpr uint8_t COLOR_ELEMENTS     = 3;
constexpr uint8_t COLOR_ELEMENT_SIZE = sizeof (uint16_t);
constexpr uint8_t COLOR_SIZE         = COLOR_ELEMENT_SIZE * 3;


template <typename coordinateType, typename colorType>
PotreeExporter<coordinateType, colorType>::PotreeExporter (
        const PointCloud& pointCloud,
        const std::shared_ptr<Chunk[]>& octree,
        const GpuArrayU32& leafeLut,
        const std::shared_ptr<SubsamplingData>& subsamples,
        OctreeMetadata metadata,
        PointCloudMetadata cloudMetadata,
        SubsampleMetadata subsamplingMetadata) :
        OctreeExporter<coordinateType, colorType> (
                pointCloud, octree, leafeLut, subsamples, metadata, cloudMetadata, subsamplingMetadata)
{}

template <typename coordinateType, typename colorType>
void PotreeExporter<coordinateType, colorType>::exportOctree (const std::string& path)
{
    this->itsPointsExported = 0;
    itsExportFolder         = path;
    itsExportedNodes        = 0;
    createBinaryHierarchyFiles ();
    createMetadataFile ();
}

template <typename coordinateType, typename colorType>
void PotreeExporter<coordinateType, colorType>::createBinaryHierarchyFiles ()
{
    std::ofstream pointFile;
    pointFile.open (itsExportFolder + POINT_FILE_NAME, std::ios::binary);
    std::ofstream hierarchyFile;
    hierarchyFile.open (itsExportFolder + HIERARCHY_FILE_NAME, std::ios::binary);

    breathFirstExport (pointFile, hierarchyFile);

    pointFile.close ();
    hierarchyFile.close ();
}

template <typename coordinateType, typename colorType>
ExportResult PotreeExporter<coordinateType, colorType>::exportNode (uint32_t nodeIndex)
{
    bool isFinished       = this->isFinishedNode (nodeIndex);
    uint64_t nodeByteSize = 0;
    uint32_t validPoints  = 0;

    std::unique_ptr<uint8_t[]> buffer;

    if (isFinished)
    {
        bool isAveraging  = this->itsSubsampleMetadata.performAveraging;
        bool isParent     = this->isParentNode (nodeIndex);
        auto pointsInNode = this->getPointsInNode (nodeIndex);
        const std::unique_ptr<uint32_t[]>& lut =
                isParent ? this->itsSubsamples->getLutHost (nodeIndex) : this->itsLeafLut;


        uint32_t dataStride = this->itsCloudMetadata.pointDataStride;

        uint64_t bufferOffset  = 0;
        uint32_t bytesPerPoint = 3 * (sizeof (int32_t) + sizeof (uint16_t));
        buffer                 = std::make_unique<uint8_t[]> (pointsInNode * bytesPerPoint);

        // Export all point to pointFile
        for (uint64_t u = 0; u < pointsInNode; ++u)
        {
            if (isParent)
            {
                if (lut[u] != INVALID_INDEX)
                {
                    ++validPoints;
                    uint64_t pointByteIndex = lut[u] * dataStride;
                    bufferOffset += writeCoordinatesBuffered (buffer, bufferOffset, pointByteIndex);

                    if (isAveraging)
                    {
                        bufferOffset += writeColorsBuffered (buffer, bufferOffset, nodeIndex, u);
                    }

                    else
                    {
                        pointByteIndex += (3 * sizeof (coordinateType));
                        bufferOffset += writeSimpleColorsBuffered (buffer, bufferOffset, pointByteIndex);
                    }
                }
            }
            else
            {
                if (lut[this->itsOctree[nodeIndex].chunkDataIndex + u] != INVALID_INDEX)
                {
                    ++validPoints;
                    uint32_t pointByteIndex = lut[this->itsOctree[nodeIndex].chunkDataIndex + u] * dataStride;
                    bufferOffset += writeCoordinatesBuffered (buffer, bufferOffset, pointByteIndex);
                    pointByteIndex += sizeof (coordinateType) * 3;
                    bufferOffset += writeSimpleColorsBuffered (buffer, bufferOffset, pointByteIndex);
                }
            }
        }
        nodeByteSize = validPoints * bytesPerPoint;
    }

    uint8_t bitmask = getChildMask (nodeIndex);
    uint8_t type    = bitmask == 0 ? 1 : 0;

    return {type, bitmask, validPoints, nodeByteSize, std::move (buffer)};
}

template <typename coordinateType, typename colorType>
void PotreeExporter<coordinateType, colorType>::breathFirstExport (
        std::ofstream& pointFile, std::ofstream& hierarchyFile)
{
    std::unordered_map<uint32_t, bool> discoveredNodes;
    std::list<uint32_t> toVisit;

    discoveredNodes[this->getRootIndex ()] = true;
    toVisit.push_back (this->getRootIndex ());

    ThreadPool pool (thread::hardware_concurrency ());

    while (!toVisit.empty ())
    {
        auto node = toVisit.front ();
        toVisit.pop_front ();

        itsFutureResults.push_back (pool.enqueue ([this, node] { return std::move (exportNode (node)); }));

        for (auto i = 0; i < 8; ++i)
        {
            int childNode = this->getChildNodeIndex (node, i);
            if (childNode != -1 && discoveredNodes.find (childNode) == discoveredNodes.end () &&
                this->isFinishedNode (childNode))
            {
                discoveredNodes[childNode] = true;
                toVisit.push_back (childNode);
            }
        }
    }

    exportBuffers (pointFile, hierarchyFile);
}

template <typename coordinateType, typename colorType>
void PotreeExporter<coordinateType, colorType>::exportBuffers (std::ofstream& pointFile, std::ofstream& hierarchyFile)
{
    uint64_t byteOffset = 0;

    // Write out result data
    for (auto i = 0; i < itsFutureResults.size (); ++i)
    {
        const ExportResult& result = itsFutureResults[i].get ();

        // Write out binary and hierarchy data
        pointFile.write (reinterpret_cast<const char*> (&(result.buffer[0])), result.nodeByteSize);
        HierarchyFileEntry entry{result.type, result.bitmask, result.validPoints, byteOffset, result.nodeByteSize};
        hierarchyFile.write (reinterpret_cast<const char*> (&entry), sizeof (HierarchyFileEntry));

        // Increase local statistics
        byteOffset += result.nodeByteSize;
        ++itsExportedNodes;
        this->itsPointsExported += result.validPoints;
    }
    spdlog::info ("Exported {} nodes / {} points", itsExportedNodes, this->itsPointsExported);
}


template <typename coordinateType, typename colorType>
inline uint8_t PotreeExporter<coordinateType, colorType>::writeCoordinatesBuffered (
        const std::unique_ptr<uint8_t[]>& buffer, uint64_t bufferOffset, uint64_t pointByteIndex)
{
    auto* point = reinterpret_cast<coordinateType*> (this->itsCloud + pointByteIndex);
    auto scale  = this->itsCloudMetadata.scale;

    auto* dst = reinterpret_cast<int32_t*> (buffer.get () + bufferOffset);
    dst[0]    = static_cast<int32_t> (std::floor (point[0] / scale.x));
    dst[1]    = static_cast<int32_t> (std::floor (point[1] / scale.y));
    dst[2]    = static_cast<int32_t> (std::floor (point[2] / scale.z));

    return 3 * sizeof (int32_t);
}

template <typename coordinateType, typename colorType>
inline uint8_t PotreeExporter<coordinateType, colorType>::writeColorsBuffered (
        const std::unique_ptr<uint8_t[]>& buffer, uint64_t bufferOffset, uint32_t nodeIndex, uint32_t pointIndex)
{
    uint64_t encoded = this->itsSubsamples->getAvgHost (nodeIndex)[pointIndex];
    auto* dst        = reinterpret_cast<uint16_t*> (buffer.get () + bufferOffset);
    dst[0]           = static_cast<uint16_t> (encoded >> 46);
    dst[1]           = static_cast<uint16_t> (encoded >> 28);
    dst[2]           = static_cast<uint16_t> (encoded >> 10);
    return 3 * sizeof (uint16_t);
}

template <typename coordinateType, typename colorType>
inline uint8_t PotreeExporter<coordinateType, colorType>::writeSimpleColorsBuffered (
        const std::unique_ptr<uint8_t[]>& buffer, uint64_t bufferOffset, uint64_t pointByteIndex)
{
    uint32_t colorSize = sizeof (colorType);

    auto* dst = reinterpret_cast<uint16_t*> (buffer.get () + bufferOffset);
    dst[0]    = static_cast<uint16_t> (this->itsCloud[pointByteIndex]);
    dst[1]    = static_cast<uint16_t> (this->itsCloud[pointByteIndex + colorSize]);
    dst[2]    = static_cast<uint16_t> (this->itsCloud[pointByteIndex + colorSize * 2]);

    return 3 * sizeof (uint16_t);
}

template <typename coordinateType, typename colorType>
inline uint8_t PotreeExporter<coordinateType, colorType>::getChildMask (uint32_t nodeIndex)
{
    uint8_t bitmask = 0;
    for (auto i = 0; i < 8; i++)
    {
        int childNodeIndex = this->getChildNodeIndex (nodeIndex, i);
        if (childNodeIndex != -1 && this->isFinishedNode (childNodeIndex))
        {
            bitmask = bitmask | (1 << i);
        }
    }
    return bitmask;
}

template <typename coordinateType, typename colorType>
void PotreeExporter<coordinateType, colorType>::createMetadataFile ()
{
    // Prepare metadata for export
    auto bbCubic = this->itsCloudMetadata.bbCubic;
    auto scale   = this->itsCloudMetadata.scale;
    auto spacing = (bbCubic.max.x - bbCubic.min.x) / this->itsSubsampleMetadata.subsamplingGrid;

    // Common metadata
    nlohmann::ordered_json metadata;
    metadata["version"]     = POTREE_DATA_VERSION;
    metadata["name"]        = "GpuPotreeConverter";
    metadata["description"] = "AIT Austrian Institute of Technology";
    metadata["points"]      = this->itsPointsExported;
    metadata["projection"]  = "";
    metadata["flags"][0]    = this->itsSubsampleMetadata.useReplacementScheme ? "REPLACING" : "ADDITIVE";
    if (this->itsSubsampleMetadata.performAveraging)
    {
        metadata["flags"][1] = "AVERAGING";
    }
    metadata["hierarchy"]["firstChunkSize"] = itsExportedNodes * HIERARCHY_NODE_BYTES;
    metadata["hierarchy"]["stepSize"]       = HIERARCHY_STEP_SIZE;
    metadata["hierarchy"]["depth"]          = HIERARCHY_DEPTH;
    metadata["offset"]                      = {0, 0, 0}; // We are not shifting the cloud
    metadata["scale"]                       = {scale.x, scale.y, scale.z};
    metadata["spacing"]                     = spacing;
    metadata["boundingBox"]["min"]          = {bbCubic.min.x, bbCubic.min.y, bbCubic.min.z};
    metadata["boundingBox"]["max"]          = {bbCubic.max.x, bbCubic.max.y, bbCubic.max.z};
    metadata["encoding"]                    = POTREE_DATA_ENCODING;

    // POSITION attribute
    metadata["attributes"][0]["name"]        = POSITION_NAME;
    metadata["attributes"][0]["description"] = "";
    metadata["attributes"][0]["size"]        = POSITION_SIZE;
    metadata["attributes"][0]["numElements"] = POSITION_ELEMENTS;
    metadata["attributes"][0]["elementSize"] = POSITION_ELEMENT_SIZE;
    metadata["attributes"][0]["type"]        = POSITION_TYPE;
    metadata["attributes"][0]["min"]         = {bbCubic.min.x, bbCubic.min.y, bbCubic.min.z};
    metadata["attributes"][0]["max"]         = {bbCubic.max.x, bbCubic.max.y, bbCubic.max.z};

    // COLOR attribute
    metadata["attributes"][1]["name"]        = COLOR_NAME;
    metadata["attributes"][1]["description"] = "";
    metadata["attributes"][1]["size"]        = COLOR_SIZE;
    metadata["attributes"][1]["numElements"] = COLOR_ELEMENTS;
    metadata["attributes"][1]["elementSize"] = COLOR_ELEMENT_SIZE;
    metadata["attributes"][1]["type"]        = COLOR_TYPE;
    metadata["attributes"][1]["min"]         = {0, 0, 0};
    metadata["attributes"][1]["max"]         = {65024, 65280, 65280};

    std::ofstream file (itsExportFolder + METADATA_FILE_NAME);
    file << std::setw (4) << metadata;
    file.close ();
}


//----------------------------------------------------------------------------------------------------------------------
//                                           SparseOctree<float, uint8_t>
//----------------------------------------------------------------------------------------------------------------------
template PotreeExporter<float, uint8_t>::PotreeExporter (
        const PointCloud& pointCloud,
        const std::shared_ptr<Chunk[]>& octree,
        const GpuArrayU32& leafeLut,
        const std::shared_ptr<SubsamplingData>& subsamples,
        OctreeMetadata metadata,
        PointCloudMetadata cloudMetadata,
        SubsampleMetadata subsamplingMetadata);

template void PotreeExporter<float, uint8_t>::exportOctree (const std::string& path);

//----------------------------------------------------------------------------------------------------------------------
//                                           SparseOctree<double, uint8_t>
//----------------------------------------------------------------------------------------------------------------------
template PotreeExporter<double, uint8_t>::PotreeExporter (
        const PointCloud& pointCloud,
        const std::shared_ptr<Chunk[]>& octree,
        const GpuArrayU32& leafeLut,
        const std::shared_ptr<SubsamplingData>& subsamples,
        OctreeMetadata metadata,
        PointCloudMetadata cloudMetadata,
        SubsampleMetadata subsamplingMetadata);

template void PotreeExporter<double, uint8_t>::exportOctree (const std::string& path);