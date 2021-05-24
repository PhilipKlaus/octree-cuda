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


PotreeExporter::PotreeExporter () : OctreeExporter ()
{}

void PotreeExporter::exportOctree (
        const std::string& path,
        const PointCloud& pointCloud,
        const Octree& octree,
        const ProcessingInfo& subsampleMetadata)
{
    this->itsPointsExported = 0;
    itsExportFolder         = path;
    itsExportedNodes        = 0;
    createBinaryHierarchyFiles (pointCloud, octree);
    createMetadataFile (pointCloud, subsampleMetadata);
}

void PotreeExporter::createBinaryHierarchyFiles (const PointCloud& cloud, const Octree& octree)
{
    std::ofstream pointFile;
    pointFile.open (itsExportFolder + POINT_FILE_NAME, std::ios::binary);
    std::ofstream hierarchyFile;
    hierarchyFile.open (itsExportFolder + HIERARCHY_FILE_NAME, std::ios::binary);

    breathFirstExport (pointFile, hierarchyFile, octree);
    pointFile.write (reinterpret_cast<const char*> (cloud->getOutputBuffer_h ().get ()), cloud->getOutputBufferSize ());

    pointFile.close ();
    hierarchyFile.close ();
}

void PotreeExporter::breathFirstExport (std::ofstream& pointFile, std::ofstream& hierarchyFile, const Octree& octree)
{
    std::unordered_map<uint32_t, bool> discoveredNodes;
    std::list<uint32_t> toVisit;

    discoveredNodes[octree->getRootIndex ()] = true;
    toVisit.push_back (octree->getRootIndex ());

    while (!toVisit.empty ())
    {
        auto node = toVisit.front ();
        toVisit.pop_front ();
        uint32_t pointsInNode = octree->getNode (node).pointCount;

        uint8_t bitmask     = getChildMask (octree, node);
        uint8_t type        = bitmask == 0 ? 1 : 0;
        uint64_t byteOffset = octree->getNode (node).dataIdx * (3 * (sizeof (uint32_t) + sizeof (uint16_t)));
        uint64_t byteSize   = pointsInNode * (3 * (sizeof (uint32_t) + sizeof (uint16_t)));
        HierarchyFileEntry entry{type, bitmask, pointsInNode, byteOffset, byteSize};
        hierarchyFile.write (reinterpret_cast<const char*> (&entry), sizeof (HierarchyFileEntry));

        this->itsPointsExported += pointsInNode;
        ++itsExportedNodes;

        for (int childNode : octree->getNode (node).childNodes)
        {
            if (childNode != -1 && discoveredNodes.find (childNode) == discoveredNodes.end () &&
                octree->getNode (childNode).isFinished)
            {
                discoveredNodes[childNode] = true;
                toVisit.push_back (childNode);
            }
        }
    }
}


inline uint8_t PotreeExporter::getChildMask (const Octree& octree, uint32_t nodeIndex)
{
    uint8_t bitmask = 0;
    for (auto i = 0; i < 8; i++)
    {
        int childNodeIndex = octree->getNode (nodeIndex).childNodes[i];
        if (childNodeIndex != -1 && octree->getNode (childNodeIndex).isFinished)
        {
            bitmask = bitmask | (1 << i);
        }
    }
    return bitmask;
}

void PotreeExporter::createMetadataFile (const PointCloud& cloud, const ProcessingInfo& subsampleMeta) const
{
    // Prepare metadata for export
    auto& cloudMeta = cloud->getMetadata ();
    auto scale      = cloudMeta.scale;
    auto sideLength = cloudMeta.cubicSize ();
    auto spacing    = (sideLength) / subsampleMeta.subsamplingGrid;

    // Common metadata
    nlohmann::ordered_json metadata;
    metadata["version"]     = POTREE_DATA_VERSION;
    metadata["name"]        = "GpuPotreeConverter";
    metadata["description"] = "AIT Austrian Institute of Technology";
    metadata["points"]      = this->itsPointsExported;
    metadata["projection"]  = "";
    metadata["flags"][0]    = subsampleMeta.useReplacementScheme ? "REPLACING" : "ADDITIVE";
    if (subsampleMeta.useAveraging)
    {
        metadata["flags"][1] = "AVERAGING";
    }
    metadata["hierarchy"]["firstChunkSize"] = itsExportedNodes * HIERARCHY_NODE_BYTES;
    metadata["hierarchy"]["stepSize"]       = HIERARCHY_STEP_SIZE;
    metadata["hierarchy"]["depth"]          = HIERARCHY_DEPTH;
    metadata["offset"]                      = {0, 0, 0}; // We are not shifting the cloud
    metadata["scale"]                       = {scale.x, scale.y, scale.z};
    metadata["spacing"]                     = spacing;
    metadata["boundingBox"]["min"]          = {0, 0, 0};
    metadata["boundingBox"]["max"]          = {sideLength, sideLength, sideLength};
    metadata["encoding"]                    = POTREE_DATA_ENCODING;

    // POSITION attribute
    metadata["attributes"][0]["name"]        = POSITION_NAME;
    metadata["attributes"][0]["description"] = "";
    metadata["attributes"][0]["size"]        = POSITION_SIZE;
    metadata["attributes"][0]["numElements"] = POSITION_ELEMENTS;
    metadata["attributes"][0]["elementSize"] = POSITION_ELEMENT_SIZE;
    metadata["attributes"][0]["type"]        = POSITION_TYPE;
    metadata["attributes"][0]["min"]         = {0, 0, 0};
    metadata["attributes"][0]["max"]         = {sideLength, sideLength, sideLength};

    // COLOR attribute
    metadata["attributes"][1]["name"]        = COLOR_NAME;
    metadata["attributes"][1]["description"] = "";
    metadata["attributes"][1]["size"]        = COLOR_SIZE;
    metadata["attributes"][1]["numElements"] = COLOR_ELEMENTS;
    metadata["attributes"][1]["elementSize"] = COLOR_ELEMENT_SIZE;
    metadata["attributes"][1]["type"]        = COLOR_TYPE;
    metadata["attributes"][1]["min"]         = {0, 0, 0};
    metadata["attributes"][1]["max"]         = {65536, 65536, 65536};

    std::ofstream file (itsExportFolder + METADATA_FILE_NAME);
    file << std::setw (4) << metadata;
    file.close ();
}
