//
// Created by KlausP on 13.11.2020.
//

#include <sparseOctree.h>
#include <json.hpp>
#include <iomanip>

using json = nlohmann::json;


template <typename coordinateType, typename colorType>
void SparseOctree<coordinateType, colorType>::calculatePointVarianceInLeafNoes(
        const unique_ptr<Chunk[]> &h_octreeSparse,
        float &sumVariance,
        float &mean,
        uint32_t nodeIndex
        ) const {

    Chunk chunk = h_octreeSparse[nodeIndex];

    // Leaf node
    if(!chunk.isParent) {
        sumVariance += pow(static_cast<float>(chunk.pointCount) - mean, 2);
    }

    // Parent node
    else {
        for(uint32_t i = 0; i < 8; ++i) {
            int childIndex = chunk.childrenChunks[i];
            if(childIndex != -1) {
                calculatePointVarianceInLeafNoes(h_octreeSparse, sumVariance, mean, childIndex);
            }
        }
    }
}


template <typename coordinateType, typename colorType>
void SparseOctree<coordinateType, colorType>::evaluateOctreeProperties(
        const unique_ptr<Chunk[]> &h_octreeSparse,
        uint32_t &leafNodes,
        uint32_t &parentNodes,
        uint32_t &pointSum,
        uint32_t &min,
        uint32_t &max,
        uint32_t nodeIndex
        ) const {

    Chunk chunk = h_octreeSparse[nodeIndex];

    // Leaf node
    if(!chunk.isParent) {
        leafNodes += 1;
        pointSum += chunk.pointCount;
        min = chunk.pointCount < min ? chunk.pointCount : min;
        max = chunk.pointCount > max ? chunk.pointCount : max;
    }

    // Parent node
    else {
        parentNodes += 1;
        for(uint32_t i = 0; i < 8; ++i) {
            int childIndex = chunk.childrenChunks[i];
            if(childIndex != -1) {
                evaluateOctreeProperties(h_octreeSparse, leafNodes, parentNodes, pointSum, min, max, chunk.childrenChunks[i]);
            }
        }
    }
}


template <typename coordinateType, typename colorType>
void SparseOctree<coordinateType, colorType>::updateOctreeStatistics() {
    // Reset Octree statistics
    itsMetadata.leafNodeAmount = 0;
    itsMetadata.parentNodeAmount = 0;
    itsMetadata.meanPointsPerLeafNode = 0.f;
    itsMetadata.stdevPointsPerLeafNode = 0.f;
    itsMetadata.minPointsPerNode = std::numeric_limits<uint32_t>::max();
    itsMetadata.maxPointsPerNode = std::numeric_limits<uint32_t>::min();

    uint32_t pointSum = 0;
    float sumVariance = 0.f;

    auto octree = getOctreeSparse();
    evaluateOctreeProperties(
            octree,
            itsMetadata.leafNodeAmount,
            itsMetadata.parentNodeAmount,
            pointSum,
            itsMetadata.minPointsPerNode,
            itsMetadata.maxPointsPerNode,
            getRootIndex()
    );
    itsMetadata.meanPointsPerLeafNode = static_cast<float>(pointSum) / itsMetadata.leafNodeAmount;

    calculatePointVarianceInLeafNoes(octree, sumVariance, itsMetadata.meanPointsPerLeafNode, getRootIndex());
    itsMetadata.stdevPointsPerLeafNode = sqrt(sumVariance / itsMetadata.leafNodeAmount);
}


template <typename coordinateType, typename colorType>
void SparseOctree<coordinateType, colorType>::exportOctreeStatistics(const string &filePath) {

    updateOctreeStatistics();

    nlohmann::ordered_json statistics;
    statistics["depth"] = itsMetadata.depth;

    statistics["chunking"]["grid"] = itsMetadata.chunkingGrid;
    statistics["chunking"]["mergingThreshold"] = itsMetadata.mergingThreshold;

    statistics["subsampling"]["grid"] = itsMetadata.subsamplingGrid;
    switch(itsMetadata.strategy) {
        case FIRST_POINT:
            statistics["subsampling"]["strategy"] = "FIRST POINT";
            break;
        default:
            statistics["subsampling"]["strategy"] = "RANDOM POINT";
            break;
    }

    statistics["resultNodes"]["octreeNodes"] = itsMetadata.leafNodeAmount + itsMetadata.parentNodeAmount;
    statistics["resultNodes"]["leafNodeAmount"] = itsMetadata.leafNodeAmount;
    statistics["resultNodes"]["parentNodeAmount"] = itsMetadata.parentNodeAmount;
    statistics["resultNodes"]["absorbedNodes"] = itsMetadata.absorbedNodes;

    statistics["overallNodes"]["sparseOctreeNodes"] = itsMetadata.nodeAmountSparse;
    statistics["overallNodes"]["denseOctreeNodes"] = itsMetadata.nodeAmountDense;
    statistics["overallNodes"]["memorySaving"] =
            (1 - (static_cast<float>(itsMetadata.nodeAmountSparse) / itsMetadata.nodeAmountDense)) * 100;

    statistics["pointDistribution"]["meanPointsPerLeafNode"] = itsMetadata.meanPointsPerLeafNode;
    statistics["pointDistribution"]["stdevPointsPerLeafNode"] = itsMetadata.stdevPointsPerLeafNode;
    statistics["pointDistribution"]["minPointsPerNode"] = itsMetadata.minPointsPerNode;
    statistics["pointDistribution"]["maxPointsPerNode"] = itsMetadata.maxPointsPerNode;

    statistics["cloud"]["pointAmount"] = itsMetadata.cloudMetadata.pointAmount;
    statistics["cloud"]["pointDataStride"] = itsMetadata.cloudMetadata.pointDataStride;
    statistics["cloud"]["boundingBox"]["min"]["x"] = itsMetadata.cloudMetadata.boundingBox.minimum.x;
    statistics["cloud"]["boundingBox"]["min"]["y"] = itsMetadata.cloudMetadata.boundingBox.minimum.y;
    statistics["cloud"]["boundingBox"]["min"]["z"] = itsMetadata.cloudMetadata.boundingBox.minimum.z;
    statistics["cloud"]["boundingBox"]["max"]["x"] = itsMetadata.cloudMetadata.boundingBox.maximum.x;
    statistics["cloud"]["boundingBox"]["max"]["y"] = itsMetadata.cloudMetadata.boundingBox.maximum.y;
    statistics["cloud"]["boundingBox"]["max"]["z"] = itsMetadata.cloudMetadata.boundingBox.maximum.z;
    statistics["cloud"]["boundingBox"]["sideLength"] =
            itsMetadata.cloudMetadata.boundingBox.maximum.x - itsMetadata.cloudMetadata.boundingBox.minimum.x;
    statistics["cloud"]["offset"]["x"] = itsMetadata.cloudMetadata.cloudOffset.x;
    statistics["cloud"]["offset"]["y"] = itsMetadata.cloudMetadata.cloudOffset.y;
    statistics["cloud"]["offset"]["z"] = itsMetadata.cloudMetadata.cloudOffset.z;
    statistics["cloud"]["scale"]["x"] = itsMetadata.cloudMetadata.scale.x;
    statistics["cloud"]["scale"]["y"] = itsMetadata.cloudMetadata.scale.y;
    statistics["cloud"]["scale"]["z"] = itsMetadata.cloudMetadata.scale.z;

    float accumulatedTime = 0;
    for (auto const& timeEntry : itsTimeMeasurement) {
        statistics["timeMeasurements"][get<0>(timeEntry)] = get<1>(timeEntry);
        accumulatedTime += get<1>(timeEntry);
    }
    statistics["timeMeasurements"]["accumulatedGPUTime"] = accumulatedTime;


    EventWatcher& watcher = EventWatcher::getInstance();
    statistics["memory"]["peak"] = watcher.getMemoryPeak();
    statistics["memory"]["reserveEvents"] = watcher.getMemoryReserveEvents();
    statistics["memory"]["cumulatedReserved"] = watcher.getCumulatedMemoryReservation();

    for (auto const& event : watcher.getMemoryEvents()) {
        statistics["memory"]["events"][get<0>(event)] = get<1>(event);
    }

    std::ofstream file(filePath);
    file << std::setw(4) << statistics;
    file.close();
}


template <typename coordinateType, typename colorType>
void SparseOctree<coordinateType, colorType>::histogramBinning(
        const unique_ptr<Chunk[]> &h_octreeSparse,
        std::vector<uint32_t> &counts,
        uint32_t min,
        uint32_t binWidth,
        uint32_t nodeIndex
        ) const {

    Chunk chunk = h_octreeSparse[nodeIndex];

    // Leaf node
    if(!chunk.isParent) {
        uint32_t bin = (chunk.pointCount - min) / binWidth;
        ++counts[bin];
    }

        // Parent node
    else {
        for(uint32_t i = 0; i < 8; ++i) {
            int childIndex = chunk.childrenChunks[i];
            if(childIndex != -1) {
                histogramBinning(h_octreeSparse, counts, min, binWidth, chunk.childrenChunks[i]);
            }
        }
    }
}


template <typename coordinateType, typename colorType>
void SparseOctree<coordinateType, colorType>::exportHistogram(const string &filePath, uint32_t binWidth) {
    updateOctreeStatistics();

    if(binWidth == 0) {
        binWidth = static_cast<uint32_t>(ceil(3.5f * (itsMetadata.stdevPointsPerLeafNode / pow(itsMetadata.leafNodeAmount, 1.f/3.f))));
    }
    auto binAmount = static_cast<uint32_t>(ceil(itsMetadata.maxPointsPerNode - itsMetadata.minPointsPerNode) / binWidth);
    std::vector<uint32_t> counts;
    for(uint32_t i = 0; i < binAmount; ++i) {
        counts.push_back(0);
    }
    histogramBinning(getOctreeSparse(), counts, itsMetadata.minPointsPerNode, binWidth, getRootIndex());

    const std::string itsHtmlPart1 =
            "<html>\n"
            "    <head>\n"
            "        <script src=\"https://cdn.jsdelivr.net/npm/chart.js@2.8.0\"></script>\n"
            "        <style>\n"
            "            html, body {\n"
            "            margin: 0;\n"
            "            height: 100%;\n"
            "            }\n"
            "            canvas {\n"
            "            width: 100%;\n"
            "            height: 100%;\n"
            "            display: block;\n"
            "            }\n"
            "        </style>\n"
            "    </head>\n"
            "    <body>\n"
            "        <canvas id=\"myChart\"></canvas>\n"
            "        <script>\n"
            "            var ctx = document.getElementById('myChart').getContext('2d');\n"
            "            var chart = new Chart(ctx, {\n"
            "                // The type of chart we want to create\n"
            "                type: 'bar',\n"
            "\n"
            "                // The data for our dataset\n"
            "                data:{";

    const std::string itsHtmlPart2 =
            "               }\n"
            "            });\n"
            "        </script>\n"
            "    </body>\n"
            "</html>";

    string labels = "labels:[";
    string data = "data:[";
    string label = "'Point Distribution: binWidth(" +
            to_string(binWidth) + "), mergingThreshold(" +
            to_string(itsMetadata.mergingThreshold) + "), points(" +
            to_string(itsMetadata.cloudMetadata.pointAmount) + ")'";

    for(uint32_t i = 0; i < binAmount; ++i) {
        labels += ("'" + to_string(itsMetadata.minPointsPerNode + i * binWidth) + " - " + to_string(itsMetadata.minPointsPerNode + (i + 1) * binWidth) + "'");
        data += to_string(counts[i]);
        if(i < (binAmount - 1)) {
            labels += ",";
            data += ",";
        }
    }
    labels += "]";
    data += "]";

    std::ofstream htmlData;
    htmlData.open (std::string(filePath), std::ios::out);
    htmlData << itsHtmlPart1;
    htmlData << labels + ", datasets:[{label: " + label + ", backgroundColor: 'rgb(255, 99, 132)', borderColor: 'rgb(255, 99, 132)', " + data + "}]";
    htmlData << itsHtmlPart2;
    htmlData.close();
};


// Template definitions for coordinateType: float, colorType: uint8_t

template void SparseOctree<float, uint8_t>::calculatePointVarianceInLeafNoes(
        const unique_ptr<Chunk[]> &h_octreeSparse,
        float &sumVariance,
        float &mean,
        uint32_t nodeIndex
) const ;

template void SparseOctree<float, uint8_t>::evaluateOctreeProperties(
        const unique_ptr<Chunk[]> &h_octreeSparse,
        uint32_t &leafNodes,
        uint32_t &parentNodes,
        uint32_t &pointSum,
        uint32_t &min,
        uint32_t &max,
        uint32_t nodeIndex
) const;

template void SparseOctree<float, uint8_t>::updateOctreeStatistics();
template void SparseOctree<float, uint8_t>::exportOctreeStatistics(const string &filePath);
template void SparseOctree<float, uint8_t>::histogramBinning(
        const unique_ptr<Chunk[]> &h_octreeSparse,
        std::vector<uint32_t> &counts,
        uint32_t min,
        uint32_t binWidth,
        uint32_t nodeIndex
) const;

template void SparseOctree<float, uint8_t>::exportHistogram(const string &filePath, uint32_t binWidth);