#include "potree_exporter.cuh"
#include <iomanip>
#include <iostream>
#include <json.hpp>
#include <queue>
#include <unordered_map>
#include <list>

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
        const GpuArrayU8& pointCloud,
        const GpuOctree& octree,
        const GpuArrayU32& leafeLut,
        const unordered_map<uint32_t, GpuArrayU32>& parentLut,
        const unordered_map<uint32_t, GpuAveraging>& parentAveraging,
        OctreeMetadata metadata) :
        OctreeExporter<coordinateType, colorType> (pointCloud, octree, leafeLut, parentLut, parentAveraging, metadata)
{}

template <typename coordinateType, typename colorType>
void PotreeExporter<coordinateType, colorType>::exportOctree (const std::string& path)
{
    this->itsPointsExported = 0;
    itsExportFolder         = path;
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
void PotreeExporter<coordinateType, colorType>::exportNode (std::promise<ExportResult> && result, uint32_t nodeIndex)
{
    bool isFinished       = this->isFinishedNode (nodeIndex);
    uint64_t nodeByteSize = 0;
    uint32_t validPoints = 0;

    std::unique_ptr<uint8_t []> buffer;

    if (isFinished)
    {
        bool isAveraging                       = true;
        bool isParent                          = this->isParentNode (nodeIndex);
        auto pointsInNode                      = this->getPointsInNode (nodeIndex);
        const std::unique_ptr<uint32_t[]>& lut = isParent ? this->itsParentLut[nodeIndex] : this->itsLeafeLut;


        uint32_t dataStride  = this->itsMetadata.cloudMetadata.pointDataStride;

        uint64_t bufferOffset = 0;
        uint32_t bytesPerPoint = 3 * (sizeof (int32_t) + sizeof (uint16_t));
        buffer = std::make_unique<uint8_t[]> (pointsInNode * bytesPerPoint);

        // Export all point to pointFile
        for (uint64_t u = 0; u < pointsInNode; ++u)
        {
            if (isParent)
            {
                if (lut[u] != INVALID_INDEX)
                {
                    ++validPoints;
                    uint64_t pointByteIndex = lut[u] * dataStride;
                    bufferOffset += writePointCoordinates (buffer, bufferOffset, pointByteIndex);

                    if (isAveraging)
                    {
                        bufferOffset += writeColorAveraged (buffer, bufferOffset, nodeIndex, u);
                    }

                    else
                    {
                        pointByteIndex += (3 * sizeof (coordinateType));
                        bufferOffset += writeColorNonAveraged (buffer, bufferOffset, pointByteIndex);
                    }
                }
            }
            else
            {
                if (lut[this->itsOctree[nodeIndex].chunkDataIndex + u] != INVALID_INDEX)
                {
                    ++validPoints;
                    uint32_t pointByteIndex = lut[this->itsOctree[nodeIndex].chunkDataIndex + u] * dataStride;
                    bufferOffset += writePointCoordinates (buffer, bufferOffset, pointByteIndex);
                    pointByteIndex += sizeof (coordinateType) * 3;
                    bufferOffset += writeColorNonAveraged (buffer, bufferOffset, pointByteIndex);
                }
            }
        }
        nodeByteSize    = validPoints * bytesPerPoint;
    }

    uint8_t bitmask = getChildMask (nodeIndex);
    uint8_t type = bitmask == 0 ? 1 : 0;

    result.set_value({
            type,
            bitmask,
            validPoints,
            nodeByteSize,
            std::move(buffer)
    });
}

template <typename coordinateType, typename colorType>
void PotreeExporter<coordinateType, colorType>::breathFirstExport (
        std::ofstream& pointFile, std::ofstream& hierarchyFile)
{
    uint32_t exportedNodes = 0;
    uint64_t byteOffset    = 0;
    std::unordered_map<uint32_t, bool> discoveredNodes;
    std::list<uint32_t> toVisit;

    discoveredNodes[this->getRootIndex ()] = true;
    toVisit.push_back (this->getRootIndex ());

    auto start = std::chrono::high_resolution_clock::now ();

    while (!toVisit.empty ())
    {
        auto node = toVisit.front ();
        toVisit.pop_front ();

        if(itsFutureResults.size() >= 4) {
            for(auto i = 0; i < itsFutureResults.size(); ++i) {
                itsThreads[i].join();
                itsResults.push_back(itsFutureResults[i].get());
            }
            itsThreads.clear();
            itsFutureResults.clear();
        }

        std::promise<ExportResult> promise;
        auto result = promise.get_future();
        auto t = std::thread(&PotreeExporter<coordinateType, colorType>::exportNode, this, std::move(promise), node);
        itsFutureResults.push_back(std::move(result));
        itsThreads.push_back(std::move(t));


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

    for(auto i = 0; i < itsFutureResults.size(); ++i) {
        itsThreads[i].join();
        itsResults.push_back(itsFutureResults[i].get());
    }
    itsThreads.clear();
    itsFutureResults.clear();

    for(auto i = 0; i < itsResults.size(); ++i) {
        const ExportResult &result = itsResults[i];

        pointFile.write (reinterpret_cast<const char*> (&(result.buffer[0])), result.nodeByteSize);

        HierarchyFileEntry entry{result.type, result.bitmask, result.validPoints, byteOffset, result.nodeByteSize};
        hierarchyFile.write (reinterpret_cast<const char*> (&entry), sizeof (HierarchyFileEntry));

        byteOffset += result.nodeByteSize;
        this->itsPointsExported += result.validPoints;
        ++exportedNodes;
    }

    auto finish                           = std::chrono::high_resolution_clock::now ();
    std::chrono::duration<double> elapsed = finish - start;

    spdlog::info ("Exported {} nodes / {} points in {} seconds", exportedNodes, this->itsPointsExported, elapsed.count ());
}

template <typename coordinateType, typename colorType>
inline uint8_t PotreeExporter<coordinateType, colorType>::writePointCoordinates (
        const std::unique_ptr<uint8_t[]>& buffer, uint64_t bufferOffset, uint64_t pointByteIndex)
{
    auto* point = reinterpret_cast<coordinateType*> (this->itsPointCloud.get () + pointByteIndex);
    auto scale  = this->itsMetadata.cloudMetadata.scale;

    auto *dst = reinterpret_cast<int32_t *>(buffer.get() + bufferOffset);
    dst[0] = static_cast<int32_t> (std::floor (point[0] / scale.x));
    dst[1] = static_cast<int32_t> (std::floor (point[1] / scale.y));
    dst[2] = static_cast<int32_t> (std::floor (point[2] / scale.z));

    return 3 * sizeof (int32_t);
}

template <typename coordinateType, typename colorType>
inline uint8_t PotreeExporter<coordinateType, colorType>::writeColorAveraged (
        const std::unique_ptr<uint8_t[]>& buffer, uint64_t bufferOffset, uint32_t nodeIndex, uint32_t pointIndex)
{
    uint32_t sumPointCount = this->itsAveraging[nodeIndex][pointIndex].pointCount;

    auto *dst = reinterpret_cast<uint16_t *>(buffer.get() + bufferOffset);
    dst[0] = static_cast<uint16_t> (this->itsAveraging[nodeIndex][pointIndex].r / sumPointCount);
    dst[1] = static_cast<uint16_t> (this->itsAveraging[nodeIndex][pointIndex].g / sumPointCount);
    dst[2] = static_cast<uint16_t> (this->itsAveraging[nodeIndex][pointIndex].b / sumPointCount);

    return 3 * sizeof (uint16_t);
}

template <typename coordinateType, typename colorType>
inline uint8_t PotreeExporter<coordinateType, colorType>::writeColorNonAveraged (
        const std::unique_ptr<uint8_t[]>& buffer, uint64_t bufferOffset, uint64_t pointByteIndex)
{
    uint32_t colorSize = sizeof (colorType);

    auto *dst = reinterpret_cast<uint16_t *>(buffer.get() + bufferOffset);
    dst[0] = static_cast<uint16_t> (this->itsPointCloud[pointByteIndex]);
    dst[1] = static_cast<uint16_t> (this->itsPointCloud[pointByteIndex + colorSize]);
    dst[2] = static_cast<uint16_t> (this->itsPointCloud[pointByteIndex + colorSize * 2]);

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
    uint32_t exportedNodes = this->itsMetadata.leafNodeAmount + this->itsMetadata.parentNodeAmount;
    auto bbReal            = this->itsMetadata.cloudMetadata.bbReal;
    auto bbCubic           = this->itsMetadata.cloudMetadata.bbCubic;
    auto scale             = this->itsMetadata.cloudMetadata.scale;
    auto spacing           = (bbCubic.max.x - bbCubic.min.x) / this->itsMetadata.subsamplingGrid;

    spdlog::info (
            "EXPORTED BB: min[x,y,z]=[{},{},{}], max[x,y,z]=[{},{},{}]",
            bbReal.min.x,
            bbReal.min.y,
            bbReal.min.z,
            bbReal.max.x,
            bbReal.max.y,
            bbReal.max.z);

    // Common metadata
    nlohmann::ordered_json metadata;
    metadata["version"]                     = POTREE_DATA_VERSION;
    metadata["name"]                        = "GpuPotreeConverter";
    metadata["description"]                 = "AIT Austrian Institute of Technology";
    metadata["points"]                      = this->itsPointsExported;
    metadata["projection"]                  = "";
    metadata["hierarchy"]["firstChunkSize"] = exportedNodes * HIERARCHY_NODE_BYTES;
    metadata["hierarchy"]["stepSize"]       = HIERARCHY_STEP_SIZE;
    metadata["hierarchy"]["depth"]          = HIERARCHY_DEPTH;
    metadata["offset"]                      = {bbReal.min.x, bbReal.min.y, bbReal.min.z}; // ToDo: real offset !!! currently 0
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
        const GpuArrayU8& pointCloud,
        const GpuOctree& octree,
        const GpuArrayU32& leafeLut,
        const unordered_map<uint32_t, GpuArrayU32>& parentLut,
        const unordered_map<uint32_t, GpuAveraging>& parentAveraging,
        OctreeMetadata metadata);

template void PotreeExporter<float, uint8_t>::exportOctree (const std::string& path);

//----------------------------------------------------------------------------------------------------------------------
//                                           SparseOctree<double, uint8_t>
//----------------------------------------------------------------------------------------------------------------------
template PotreeExporter<double, uint8_t>::PotreeExporter (
        const GpuArrayU8& pointCloud,
        const GpuOctree& octree,
        const GpuArrayU32& leafeLut,
        const unordered_map<uint32_t, GpuArrayU32>& parentLut,
        const unordered_map<uint32_t, GpuAveraging>& parentAveraging,
        OctreeMetadata metadata);

template void PotreeExporter<double, uint8_t>::exportOctree (const std::string& path);