#include "potree_exporter.cuh"
#include <iomanip>
#include <iostream>
#include <json.hpp>
#include <queue>
#include <unordered_map>

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
    pointFile.open (itsExportFolder + R"(/octree.bin)", std::ios::binary);
    std::ofstream hierarchyFile;
    hierarchyFile.open (itsExportFolder + R"(/hierarchy.bin)", std::ios::binary);

    //exportNode(this->getRootIndex(), 0, pointFile, hierarchyFile);
    breathFirstExport(pointFile, hierarchyFile);

    pointFile.close();
    hierarchyFile.close();
}

template <typename coordinateType, typename colorType>
uint64_t PotreeExporter<coordinateType, colorType>::exportNode (
        uint32_t nodeIndex, uint64_t byteOffset, std::ofstream& pointFile, std::ofstream& hierarchyFile)
{
    bool isFinished = this->isFinishedNode (nodeIndex);
    uint64_t nodeByteSize = 0;

    if(isFinished) {
        bool isAveraging = true;
        bool isParent   = this->isParentNode (nodeIndex);
        auto pointsInNode = this->getPointsInNode (nodeIndex);
        const std::unique_ptr<uint32_t[]>& lut = isParent ? this->itsParentLut[nodeIndex] : this->itsLeafeLut;

        uint32_t validPoints = 0;
        uint32_t dataStride = this->itsMetadata.cloudMetadata.pointDataStride;

        // Export all point to pointFile
        for (uint64_t u = 0; u < pointsInNode; ++u)
        {
            if (isParent)
            {
                if (lut[u] != INVALID_INDEX)
                {
                    ++validPoints;
                    uint64_t pointByteIndex = lut[u] * dataStride;
                    writePointCoordinates (pointFile, pointByteIndex);

                    if (isAveraging)
                    {
                        writeColorAveraged (pointFile, nodeIndex, u);
                    }

                    else
                    {
                        writeColorNonAveraged (pointFile, pointByteIndex);
                    }
                }
            }
            else
            {
                if (lut[this->itsOctree[nodeIndex].chunkDataIndex + u] != INVALID_INDEX)
                {
                    ++validPoints;
                    uint32_t pointByteIndex = lut[this->itsOctree[nodeIndex].chunkDataIndex + u] * dataStride;
                    writePointCoordinates (pointFile, pointByteIndex);
                    pointByteIndex += sizeof (coordinateType) * 3;
                    writeColorNonAveraged (pointFile, pointByteIndex);
                }
            }
        }
        this->itsPointsExported += validPoints;

        // Export hierarchy to hierarchyFile
        uint8_t bitmask = getChildMask(nodeIndex);
        uint8_t type = bitmask == 0 ? 1 : 0;
        nodeByteSize = validPoints * 3 * (sizeof(int32_t) + sizeof (uint16_t));

        hierarchyFile.write(reinterpret_cast<const char*>(&type), sizeof (uint8_t));
        hierarchyFile.write(reinterpret_cast<const char*>(&bitmask), sizeof (uint8_t));
        hierarchyFile.write(reinterpret_cast<const char*>(&validPoints), sizeof (uint32_t));
        hierarchyFile.write(reinterpret_cast<const char*>(&byteOffset), sizeof (uint64_t));
        hierarchyFile.write(reinterpret_cast<const char*>(&nodeByteSize), sizeof (uint64_t));
    }
    return nodeByteSize;
}

template <typename coordinateType, typename colorType>
void PotreeExporter<coordinateType, colorType>::breathFirstExport (std::ofstream& pointFile, std::ofstream& hierarchyFile)
{
    uint32_t exportedNodes = 0;
    uint64_t byteOffset = 0;
    std::unordered_map<uint32_t, bool> discoveredNodes;
    std::queue<uint32_t> toVisit;

    discoveredNodes[this->getRootIndex()] = true;
    toVisit.push(this->getRootIndex());

    auto start = std::chrono::high_resolution_clock::now();

    while(!toVisit.empty()) {
        auto node = toVisit.front();
        toVisit.pop();
        byteOffset += exportNode(node, byteOffset, pointFile, hierarchyFile);
        ++exportedNodes;

        for(auto i = 0; i < 8; ++i) {
            int childNode = this->getChildNodeIndex (node, i);
            if (childNode != -1 && discoveredNodes.find(childNode) == discoveredNodes.end() && this->isFinishedNode(childNode))
            {
                discoveredNodes[childNode] = true;
                toVisit.push(childNode);
            }
        }
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;

    spdlog::info("Exported {} nodes in {} seconds", exportedNodes, elapsed.count());
}

// ToDo: divide by scale
template <typename coordinateType, typename colorType>
void PotreeExporter<coordinateType, colorType>::writePointCoordinates (
        std::ofstream & pointFile, uint64_t pointByteIndex)
{
    auto* point = reinterpret_cast<Vector3<coordinateType>*> (this->itsPointCloud.get() + pointByteIndex);
    int32_t coords[3];
    coords[0] = static_cast<int32_t>(floor(point->x / 0.001));
    coords[1] = static_cast<int32_t>(floor(point->y / 0.001));
    coords[2] = static_cast<int32_t>(floor(point->z/ 0.001));

    pointFile.write(reinterpret_cast<const char*>(&coords), 3 * sizeof (int32_t));
}

template <typename coordinateType, typename colorType>
void PotreeExporter<coordinateType, colorType>::writeColorAveraged (std::ofstream & pointFile, uint32_t nodeIndex, uint32_t pointIndex)
{
    uint32_t sumPointCount = this->itsAveraging[nodeIndex][pointIndex].pointCount;

    uint16_t colors[3];
    colors[0] = static_cast<uint16_t> (this->itsAveraging[nodeIndex][pointIndex].r / sumPointCount);
    colors[1] = static_cast<uint16_t> (this->itsAveraging[nodeIndex][pointIndex].g / sumPointCount);
    colors[2] = static_cast<uint16_t> (this->itsAveraging[nodeIndex][pointIndex].b / sumPointCount);

    pointFile.write(reinterpret_cast<const char*>(&colors), 3 * sizeof (uint16_t));
}

template <typename coordinateType, typename colorType>
void PotreeExporter<coordinateType, colorType>::writeColorNonAveraged (std::ofstream & pointFile, uint64_t pointByteIndex)
{
    uint32_t colorSize = sizeof (colorType);

    uint16_t colors[3];
    colors[0] = static_cast<uint16_t >(this->itsPointCloud[pointByteIndex]);
    colors[1] = static_cast<uint16_t >(this->itsPointCloud[pointByteIndex + colorSize]);
    colors[2] = static_cast<uint16_t >(this->itsPointCloud[pointByteIndex + colorSize * 2]);

    pointFile.write(reinterpret_cast<const char*>(&colors), 3 * sizeof (uint16_t));
}

template <typename coordinateType, typename colorType>
uint8_t PotreeExporter<coordinateType, colorType>::getChildMask (uint32_t nodeIndex)
{
    uint8_t bitmask = 0;
    for(auto i = 0; i < 8; i++){
        int childNodeIndex = this->getChildNodeIndex (nodeIndex, i);
        if (childNodeIndex != -1 && this->isFinishedNode(childNodeIndex))
        {
            bitmask = bitmask | (1 << i);
        }
    }
    return bitmask;
}

template <typename coordinateType, typename colorType>
void PotreeExporter<coordinateType, colorType>::createMetadataFile ()
{
    nlohmann::ordered_json metadata;
    metadata["version"]     = "2.0";
    metadata["name"]        = "GpuPotreeConverter";
    metadata["description"] = "AIT Austrian Institute of Technology";
    metadata["points"]      = this->itsPointsExported;
    metadata["projection"]  = "";
    metadata["hierarchy"]["firstChunkSize"] =
            (this->itsMetadata.leafNodeAmount + this->itsMetadata.parentNodeAmount) * 22;
    metadata["hierarchy"]["stepSize"] = 100;
    metadata["hierarchy"]["depth"]    = 20;

    auto offset = this->itsMetadata.cloudMetadata.cloudOffset; // ToDo: evtl. 0, 0, 0
    metadata["offset"].push_back (offset.x);
    metadata["offset"].push_back (offset.y);
    metadata["offset"].push_back (offset.z);

    auto scale = this->itsMetadata.cloudMetadata.scale;
    metadata["scale"].push_back (0.001);
    metadata["scale"].push_back (0.001);
    metadata["scale"].push_back (0.001);

    auto spacing = (this->itsMetadata.cloudMetadata.boundingBox.maximum.x -
                    this->itsMetadata.cloudMetadata.boundingBox.minimum.x) /
                   this->itsMetadata.subsamplingGrid;
    metadata["spacing"] = spacing;

    auto bb = this->itsMetadata.cloudMetadata.boundingBox;
    metadata["boundingBox"]["min"].push_back (bb.minimum.x);
    metadata["boundingBox"]["min"].push_back (bb.minimum.y);
    metadata["boundingBox"]["min"].push_back (bb.minimum.z);
    metadata["boundingBox"]["max"].push_back (bb.maximum.x);
    metadata["boundingBox"]["max"].push_back (bb.maximum.y);
    metadata["boundingBox"]["max"].push_back (bb.maximum.z);

    metadata["encoding"] = "DEFAULT";


    metadata["attributes"][0]["name"]        = "position";
    metadata["attributes"][0]["description"] = "";
    metadata["attributes"][0]["size"]        = sizeof (coordinateType) * 3; // ToDo: check if correct
    metadata["attributes"][0]["numElements"] = 3;
    metadata["attributes"][0]["elementSize"] = 4;
    metadata["attributes"][0]["type"]        = "int32"; // ToDo: from config
    metadata["attributes"][0]["min"].push_back (bb.minimum.x);
    metadata["attributes"][0]["min"].push_back (bb.minimum.y);
    metadata["attributes"][0]["min"].push_back (bb.minimum.z);
    metadata["attributes"][0]["max"].push_back (bb.maximum.x);
    metadata["attributes"][0]["max"].push_back (bb.maximum.y);
    metadata["attributes"][0]["max"].push_back (bb.maximum.z);

    metadata["attributes"][1]["name"]        = "rgb";
    metadata["attributes"][1]["description"] = "";
    metadata["attributes"][1]["size"]        = 6;
    metadata["attributes"][1]["numElements"] = 3;
    metadata["attributes"][1]["elementSize"] = 2;
    metadata["attributes"][1]["type"]        = "uint16";
    metadata["attributes"][1]["min"].push_back (0);
    metadata["attributes"][1]["min"].push_back (0);
    metadata["attributes"][1]["min"].push_back (0);
    metadata["attributes"][1]["max"].push_back (65024);
    metadata["attributes"][1]["max"].push_back (65280);
    metadata["attributes"][1]["max"].push_back (65280);


    std::ofstream file (itsExportFolder + R"(/metadata.json)");
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