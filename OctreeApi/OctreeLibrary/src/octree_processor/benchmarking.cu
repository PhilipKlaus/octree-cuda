//
// Created by KlausP on 13.11.2020.
//

#include "octree_processor.h"


void OctreeProcessor::calculatePointVarianceInLeafNoes (
        const unique_ptr<Chunk[]>& h_octreeSparse, float& sumVariance, float& mean, uint32_t nodeIndex) const
{
    Chunk chunk = h_octreeSparse[nodeIndex];

    // Leaf node
    if (!chunk.isParent)
    {
        sumVariance += pow (static_cast<float> (chunk.pointCount) - mean, 2.f);
    }

    // Parent node
    else
    {
        for (uint32_t i = 0; i < 8; ++i)
        {
            int childIndex = chunk.childrenChunks[i];
            if (childIndex != -1)
            {
                calculatePointVarianceInLeafNoes (h_octreeSparse, sumVariance, mean, childIndex);
            }
        }
    }
}


void OctreeProcessor::evaluateOctreeProperties (
        const unique_ptr<Chunk[]>& h_octreeSparse,
        uint32_t& leafNodes,
        uint32_t& parentNodes,
        uint32_t& pointSum,
        uint32_t& min,
        uint32_t& max,
        uint32_t nodeIndex) const
{
    Chunk chunk = h_octreeSparse[nodeIndex];

    // Leaf node
    if (!chunk.isParent)
    {
        leafNodes += 1;
        pointSum += chunk.pointCount;
        min = chunk.pointCount < min ? chunk.pointCount : min;
        max = chunk.pointCount > max ? chunk.pointCount : max;
    }

    // Parent node
    else
    {
        parentNodes += 1;
        for (uint32_t i = 0; i < 8; ++i)
        {
            int childIndex = chunk.childrenChunks[i];
            if (childIndex != -1)
            {
                evaluateOctreeProperties (
                        h_octreeSparse, leafNodes, parentNodes, pointSum, min, max, chunk.childrenChunks[i]);
            }
        }
    }
}


void OctreeProcessor::updateOctreeStatistics ()
{
    // Reset Octree statistics
    itsMetadata.leafNodeAmount         = 0;
    itsMetadata.parentNodeAmount       = 0;
    itsMetadata.meanPointsPerLeafNode  = 0.f;
    itsMetadata.stdevPointsPerLeafNode = 0.f;
    itsMetadata.minPointsPerNode       = std::numeric_limits<uint32_t>::max ();
    itsMetadata.maxPointsPerNode       = std::numeric_limits<uint32_t>::min ();

    uint32_t pointSum = 0;
    float sumVariance = 0.f;

    auto octree = getOctreeSparse ();
    evaluateOctreeProperties (
            octree,
            itsMetadata.leafNodeAmount,
            itsMetadata.parentNodeAmount,
            pointSum,
            itsMetadata.minPointsPerNode,
            itsMetadata.maxPointsPerNode,
            getRootIndex ());
    itsMetadata.meanPointsPerLeafNode = static_cast<float> (pointSum) / itsMetadata.leafNodeAmount;

    calculatePointVarianceInLeafNoes (octree, sumVariance, itsMetadata.meanPointsPerLeafNode, getRootIndex ());
    itsMetadata.stdevPointsPerLeafNode = sqrt (sumVariance / itsMetadata.leafNodeAmount);
}


void OctreeProcessor::histogramBinning (
        const unique_ptr<Chunk[]>& h_octreeSparse,
        std::vector<uint32_t>& counts,
        uint32_t min,
        uint32_t binWidth,
        uint32_t nodeIndex) const
{
    Chunk chunk = h_octreeSparse[nodeIndex];

    // Leaf node
    if (!chunk.isParent)
    {
        uint32_t bin = (chunk.pointCount - min) / binWidth;
        ++counts[bin];
    }

    // Parent node
    else
    {
        for (uint32_t i = 0; i < 8; ++i)
        {
            int childIndex = chunk.childrenChunks[i];
            if (childIndex != -1)
            {
                histogramBinning (h_octreeSparse, counts, min, binWidth, chunk.childrenChunks[i]);
            }
        }
    }
}


void OctreeProcessor::exportHistogram (const string& filePath, uint32_t binWidth)
{
    updateOctreeStatistics ();

    if (binWidth == 0)
    {
        binWidth = static_cast<uint32_t> (
                ceil (3.5f * (itsMetadata.stdevPointsPerLeafNode / pow (itsMetadata.leafNodeAmount, 1.f / 3.f))));
    }
    auto binAmount =
            static_cast<uint32_t> (ceil (itsMetadata.maxPointsPerNode - itsMetadata.minPointsPerNode) / binWidth);
    std::vector<uint32_t> counts;
    for (uint32_t i = 0; i < binAmount; ++i)
    {
        counts.push_back (0);
    }
    histogramBinning (getOctreeSparse (), counts, itsMetadata.minPointsPerNode, binWidth, getRootIndex ());

    const std::string itsHtmlPart1 = "<html>\n"
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

    const std::string itsHtmlPart2 = "               }\n"
                                     "            });\n"
                                     "        </script>\n"
                                     "    </body>\n"
                                     "</html>";

    string labels = "labels:[";
    string data   = "data:[";
    string label  = "'Point Distribution: binWidth(" + to_string (binWidth) + "), mergingThreshold(" +
                   to_string (itsMetadata.mergingThreshold) + "), points(" +
                   to_string (itsMetadata.cloudMetadata.pointAmount) + ")'";

    for (uint32_t i = 0; i < binAmount; ++i)
    {
        labels +=
                ("'" + to_string (itsMetadata.minPointsPerNode + i * binWidth) + " - " +
                 to_string (itsMetadata.minPointsPerNode + (i + 1) * binWidth) + "'");
        data += to_string (counts[i]);
        if (i < (binAmount - 1))
        {
            labels += ",";
            data += ",";
        }
    }
    labels += "]";
    data += "]";

    std::ofstream htmlData;
    htmlData.open (std::string (filePath), std::ios::out);
    htmlData << itsHtmlPart1;
    htmlData << labels + ", datasets:[{label: " + label +
                        ", backgroundColor: 'rgb(255, 99, 132)', borderColor: 'rgb(255, 99, 132)', " + data + "}]";
    htmlData << itsHtmlPart2;
    htmlData.close ();
};
