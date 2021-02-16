//
// Created by KlausP on 25.01.2021.
//

#pragma once

#include "eventWatcher.h"
#include "metadata.h"
#include <iomanip>
#include <json.hpp>


void export_json_data (
        const std::string filePath,
        const OctreeMetadata& metadata,
        const PointCloudMetadata& cloudMetadata,
        const SubsampleMetadata& subsampleMetadata,
        const std::vector<std::tuple<std::string, float>>& timings)
{
    nlohmann::ordered_json statistics;
    statistics["depth"] = metadata.depth;

    statistics["chunking"]["grid"]             = metadata.chunkingGrid;
    statistics["chunking"]["mergingThreshold"] = metadata.mergingThreshold;

    statistics["subsampling"]["grid"]     = subsampleMetadata.subsamplingGrid;
    statistics["subsampling"]["strategy"] = "RANDOM POINT";

    statistics["resultNodes"]["octreeNodes"]      = metadata.leafNodeAmount + metadata.parentNodeAmount;
    statistics["resultNodes"]["leafNodeAmount"]   = metadata.leafNodeAmount;
    statistics["resultNodes"]["parentNodeAmount"] = metadata.parentNodeAmount;
    statistics["resultNodes"]["absorbedNodes"]    = metadata.absorbedNodes;

    statistics["overallNodes"]["sparseOctreeNodes"] = metadata.nodeAmountSparse;
    statistics["overallNodes"]["denseOctreeNodes"]  = metadata.nodeAmountDense;
    statistics["overallNodes"]["memorySaving"] =
            (1 - (static_cast<float> (metadata.nodeAmountSparse) / metadata.nodeAmountDense)) * 100;

    statistics["pointDistribution"]["meanPointsPerLeafNode"]  = metadata.meanPointsPerLeafNode;
    statistics["pointDistribution"]["stdevPointsPerLeafNode"] = metadata.stdevPointsPerLeafNode;
    statistics["pointDistribution"]["minPointsPerNode"]       = metadata.minPointsPerNode;
    statistics["pointDistribution"]["maxPointsPerNode"]       = metadata.maxPointsPerNode;

    statistics["cloud"]["pointAmount"]           = cloudMetadata.pointAmount;
    statistics["cloud"]["pointDataStride"]       = cloudMetadata.pointDataStride;
    statistics["cloud"]["bbCubic"]["min"]["x"]   = cloudMetadata.bbCubic.min.x;
    statistics["cloud"]["bbCubic"]["min"]["y"]   = cloudMetadata.bbCubic.min.y;
    statistics["cloud"]["bbCubic"]["min"]["z"]   = cloudMetadata.bbCubic.min.z;
    statistics["cloud"]["bbCubic"]["max"]["x"]   = cloudMetadata.bbCubic.max.x;
    statistics["cloud"]["bbCubic"]["max"]["y"]   = cloudMetadata.bbCubic.max.y;
    statistics["cloud"]["bbCubic"]["max"]["z"]   = cloudMetadata.bbCubic.max.z;
    statistics["cloud"]["bbCubic"]["sideLength"] = cloudMetadata.bbCubic.max.x - cloudMetadata.bbCubic.min.x;
    statistics["cloud"]["offset"]["x"]           = cloudMetadata.cloudOffset.x;
    statistics["cloud"]["offset"]["y"]           = cloudMetadata.cloudOffset.y;
    statistics["cloud"]["offset"]["z"]           = cloudMetadata.cloudOffset.z;
    statistics["cloud"]["scale"]["x"]            = cloudMetadata.scale.x;
    statistics["cloud"]["scale"]["y"]            = cloudMetadata.scale.y;
    statistics["cloud"]["scale"]["z"]            = cloudMetadata.scale.z;

    float accumulatedTime = 0;
    for (auto const& timeEntry : timings)
    {
        statistics["timeMeasurements"][std::get<0> (timeEntry)] = std::get<1> (timeEntry);
        accumulatedTime += std::get<1> (timeEntry);
    }
    statistics["timeMeasurements"]["accumulatedGPUTime"] = accumulatedTime;


    EventWatcher& watcher                     = EventWatcher::getInstance ();
    statistics["memory"]["peak"]              = watcher.getMemoryPeak ();
    statistics["memory"]["reserveEvents"]     = watcher.getMemoryReserveEvents ();
    statistics["memory"]["cumulatedReserved"] = watcher.getCumulatedMemoryReservation ();

    for (auto const& event : watcher.getMemoryEvents ())
    {
        statistics["memory"]["events"][std::get<0> (event)] = std::get<1> (event);
    }

    std::ofstream file (filePath);
    file << std::setw (4) << statistics;
    file.close ();
}