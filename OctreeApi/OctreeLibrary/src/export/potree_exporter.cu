#include "potree_exporter.cuh"
#include <iomanip>
#include <iostream>
#include <json.hpp>
#include <queue>
#include <unordered_map>

constexpr char METADATA_FILE_NAME[]    = "//metadata.json";
constexpr char HIERARCHY_FILE_NAME[]   = "//hierarchy.bin";
constexpr char POINT_FILE_NAME[]       = "//octree.bin";
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
        OctreeMetadata<coordinateType> metadata) :
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
uint64_t PotreeExporter<coordinateType, colorType>::exportNode (
        uint32_t nodeIndex, uint64_t byteOffset, std::ofstream& pointFile, std::ofstream& hierarchyFile)
{
    bool isFinished       = this->isFinishedNode (nodeIndex);
    uint64_t nodeByteSize = 0;

    if (isFinished)
    {
        bool isAveraging                       = true;
        bool isParent                          = this->isParentNode (nodeIndex);
        auto pointsInNode                      = this->getPointsInNode (nodeIndex);
        const std::unique_ptr<uint32_t[]>& lut = isParent ? this->itsParentLut[nodeIndex] : this->itsLeafeLut;

        uint32_t validPoints = 0;
        uint32_t dataStride  = this->itsMetadata.cloudMetadata.pointDataStride;

        uint64_t bufferOffset = 0;
        auto buffer = std::make_unique<uint8_t[]> (pointsInNode * (3 * (sizeof (int32_t) + sizeof (uint16_t))));

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
        this->itsPointsExported += validPoints;

        uint8_t bitmask = getChildMask (nodeIndex);
        uint8_t type    = bitmask == 0 ? 1 : 0;
        nodeByteSize    = validPoints * 3 * (sizeof (int32_t) + sizeof (uint16_t));

        // Export buffered coordinates and colors
        pointFile.write (reinterpret_cast<const char*> (&buffer[0]), nodeByteSize);

        // Export hierarchy to hierarchyFile
        HierarchyFileEntry entry{type, bitmask, validPoints, byteOffset, nodeByteSize};
        hierarchyFile.write (reinterpret_cast<const char*> (&entry), sizeof (HierarchyFileEntry));
    }
    return nodeByteSize;
}

template <typename coordinateType, typename colorType>
void PotreeExporter<coordinateType, colorType>::breathFirstExport (
        std::ofstream& pointFile, std::ofstream& hierarchyFile)
{
    uint32_t exportedNodes = 0;
    uint64_t byteOffset    = 0;
    std::unordered_map<uint32_t, bool> discoveredNodes;
    std::queue<uint32_t> toVisit;

    discoveredNodes[this->getRootIndex ()] = true;
    toVisit.push (this->getRootIndex ());

    auto start = std::chrono::high_resolution_clock::now ();

    while (!toVisit.empty ())
    {
        auto node = toVisit.front ();
        toVisit.pop ();
        byteOffset += exportNode (node, byteOffset, pointFile, hierarchyFile);
        ++exportedNodes;

        for (auto i = 0; i < 8; ++i)
        {
            int childNode = this->getChildNodeIndex (node, i);
            if (childNode != -1 && discoveredNodes.find (childNode) == discoveredNodes.end () &&
                this->isFinishedNode (childNode))
            {
                discoveredNodes[childNode] = true;
                toVisit.push (childNode);
            }
        }
    }

    auto finish                           = std::chrono::high_resolution_clock::now ();
    std::chrono::duration<double> elapsed = finish - start;

    spdlog::info ("Exported {} nodes in {} seconds", exportedNodes, elapsed.count ());
}

// ToDo: divide by scale
template <typename coordinateType, typename colorType>
uint8_t PotreeExporter<coordinateType, colorType>::writePointCoordinates (
        const std::unique_ptr<uint8_t[]>& buffer, uint64_t bufferOffset, uint64_t pointByteIndex)
{
    auto* point = reinterpret_cast<Vector3<coordinateType>*> (this->itsPointCloud.get () + pointByteIndex);
    auto realBB = this->itsMetadata.cloudMetadata.bbReal;
    auto scale  = this->itsMetadata.cloudMetadata.scale;

    uint8_t byteAmount = 3 * sizeof (int32_t);
    int32_t coords[3];
    coords[0] = static_cast<int32_t> (floor (point->x / scale.x));
    coords[1] = static_cast<int32_t> (floor (point->y / scale.y));
    coords[2] = static_cast<int32_t> (floor (point->z / scale.z));

    std::memcpy (buffer.get () + bufferOffset, &coords, byteAmount);
    return byteAmount;
}

template <typename coordinateType, typename colorType>
uint8_t PotreeExporter<coordinateType, colorType>::writeColorAveraged (
        const std::unique_ptr<uint8_t[]>& buffer, uint64_t bufferOffset, uint32_t nodeIndex, uint32_t pointIndex)
{
    uint32_t sumPointCount = this->itsAveraging[nodeIndex][pointIndex].pointCount;

    uint8_t byteAmount = 3 * sizeof (uint16_t);
    uint16_t colors[3];
    colors[0] = static_cast<uint16_t> (this->itsAveraging[nodeIndex][pointIndex].r / sumPointCount);
    colors[1] = static_cast<uint16_t> (this->itsAveraging[nodeIndex][pointIndex].g / sumPointCount);
    colors[2] = static_cast<uint16_t> (this->itsAveraging[nodeIndex][pointIndex].b / sumPointCount);

    std::memcpy (buffer.get () + bufferOffset, &colors, byteAmount);
    return byteAmount;
}

template <typename coordinateType, typename colorType>
uint8_t PotreeExporter<coordinateType, colorType>::writeColorNonAveraged (
        const std::unique_ptr<uint8_t[]>& buffer, uint64_t bufferOffset, uint64_t pointByteIndex)
{
    uint32_t colorSize = sizeof (colorType);

    uint8_t byteAmount = 3 * sizeof (uint16_t);
    uint16_t colors[3];
    colors[0] = static_cast<uint16_t> (this->itsPointCloud[pointByteIndex]);
    colors[1] = static_cast<uint16_t> (this->itsPointCloud[pointByteIndex + colorSize]);
    colors[2] = static_cast<uint16_t> (this->itsPointCloud[pointByteIndex + colorSize * 2]);

    std::memcpy (buffer.get () + bufferOffset, &colors, byteAmount);
    return byteAmount;
}

template <typename coordinateType, typename colorType>
uint8_t PotreeExporter<coordinateType, colorType>::getChildMask (uint32_t nodeIndex)
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
    metadata["offset"]                      = {bbReal.min.x, bbReal.min.y, bbReal.min.z};
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
        OctreeMetadata<float> metadata);

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
        OctreeMetadata<double> metadata);

template void PotreeExporter<double, uint8_t>::exportOctree (const std::string& path);