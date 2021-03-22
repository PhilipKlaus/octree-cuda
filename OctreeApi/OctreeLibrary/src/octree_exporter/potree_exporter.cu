#include "potree_exporter.cuh"
#include <iomanip>
#include <iostream>
#include <json.hpp>
#include <list>
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


PotreeExporter::PotreeExporter (
        const PointCloud& pointCloud,
        const std::shared_ptr<Chunk[]>& octree,
        OctreeMetadata metadata,
        PointCloudMetadata cloudMetadata,
        SubsampleMetadata subsamplingMetadata) :
        OctreeExporter (pointCloud, octree, metadata, cloudMetadata, subsamplingMetadata)
{}

void PotreeExporter::exportOctree (const std::string& path)
{
    this->itsPointsExported = 0;
    itsExportFolder         = path;
    itsExportedNodes        = 0;
    createBinaryHierarchyFiles ();
    createMetadataFile ();
}

void PotreeExporter::createBinaryHierarchyFiles ()
{
    std::ofstream pointFile;
    pointFile.open (itsExportFolder + POINT_FILE_NAME, std::ios::binary);
    std::ofstream hierarchyFile;
    hierarchyFile.open (itsExportFolder + HIERARCHY_FILE_NAME, std::ios::binary);

    breathFirstExport (pointFile, hierarchyFile);
    pointFile.write (reinterpret_cast<const char*> (this->itsOutputBuffer.get ()), this->itsOutputBufferSize);

    pointFile.close ();
    hierarchyFile.close ();
}

void PotreeExporter::breathFirstExport (std::ofstream& pointFile, std::ofstream& hierarchyFile)
{
    std::unordered_map<uint32_t, bool> discoveredNodes;
    std::list<uint32_t> toVisit;

    discoveredNodes[this->getRootIndex ()] = true;
    toVisit.push_back (this->getRootIndex ());

    while (!toVisit.empty ())
    {
        auto node = toVisit.front ();
        toVisit.pop_front ();

        uint8_t bitmask     = getChildMask (node);
        uint8_t type        = bitmask == 0 ? 1 : 0;
        uint64_t byteOffset = this->getDataIndex (node) * (3 * (sizeof (uint32_t) + sizeof (uint16_t)));
        uint64_t byteSize   = this->getPointsInNode (node) * (3 * (sizeof (uint32_t) + sizeof (uint16_t)));
        HierarchyFileEntry entry{type, bitmask, this->getPointsInNode (node), byteOffset, byteSize};
        hierarchyFile.write (reinterpret_cast<const char*> (&entry), sizeof (HierarchyFileEntry));

        this->itsPointsExported += this->getPointsInNode (node);
        ++itsExportedNodes;

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
}


inline uint8_t PotreeExporter::getChildMask (uint32_t nodeIndex)
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

void PotreeExporter::createMetadataFile ()
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
