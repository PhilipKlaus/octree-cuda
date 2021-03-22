//
// Created by KlausP on 13.11.2020.
//

#include "octree_processor_impl.cuh"

void OctreeProcessor::OctreeProcessorImpl::updateOctreeStatistics ()
{
    itsOctree->updateNodeStatistics ();
}


void OctreeProcessor::OctreeProcessorImpl::histogramBinning (
        const shared_ptr<Chunk[]>& h_octreeSparse,
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


void OctreeProcessor::OctreeProcessorImpl::exportHistogram (const string& filePath, uint32_t binWidth)
{
    updateOctreeStatistics ();
    auto& statistics = itsOctree->getNodeStatistics ();

    if (binWidth == 0)
    {
        binWidth = static_cast<uint32_t> (
                ceil (3.5f * (statistics.stdevPointsPerLeafNode / pow (statistics.leafNodeAmount, 1.f / 3.f))));
    }
    auto binAmount =
            static_cast<uint32_t> (ceil (statistics.maxPointsPerNode - statistics.minPointsPerNode) / binWidth);
    std::vector<uint32_t> counts;
    for (uint32_t i = 0; i < binAmount; ++i)
    {
        counts.push_back (0);
    }
    histogramBinning (itsOctree->getHost (), counts, statistics.minPointsPerNode, binWidth, itsOctree->getRootIndex ());

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
                   to_string (itsOctree->getMetadata ().mergingThreshold) + "), points(" +
                   to_string (itsCloud->getMetadata ().pointAmount) + ")'";

    for (uint32_t i = 0; i < binAmount; ++i)
    {
        labels +=
                ("'" + to_string (statistics.minPointsPerNode + i * binWidth) + " - " +
                 to_string (statistics.minPointsPerNode + (i + 1) * binWidth) + "'");
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
