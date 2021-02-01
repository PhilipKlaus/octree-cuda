//
// Created by KlausP on 25.01.2021.
//

#pragma once

#include "octree_metadata.h"
#include <iomanip>
#include <json.hpp>
#include "eventWatcher.h"


void export_json_data (
        const std::string filePath,
        OctreeMetadata metadata,
        const std::vector<std::tuple<std::string, float>>& timings)
{
    nlohmann::ordered_json statistics;
    statistics["depth"] = metadata.depth;

    statistics["chunking"]["grid"]             = metadata.chunkingGrid;
    statistics["chunking"]["mergingThreshold"] = metadata.mergingThreshold;

    statistics["subsampling"]["grid"] = metadata.subsamplingGrid;
    switch (metadata.strategy)
    {
    case FIRST_POINT:
        statistics["subsampling"]["strategy"] = "FIRST POINT";
        break;
    default:
        statistics["subsampling"]["strategy"] = "RANDOM POINT";
        break;
    }

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

    statistics["cloud"]["pointAmount"]         = metadata.cloudMetadata.pointAmount;
    statistics["cloud"]["pointDataStride"]     = metadata.cloudMetadata.pointDataStride;
    statistics["cloud"]["bbCubic"]["min"]["x"] = metadata.cloudMetadata.bbCubic.min.x;
    statistics["cloud"]["bbCubic"]["min"]["y"] = metadata.cloudMetadata.bbCubic.min.y;
    statistics["cloud"]["bbCubic"]["min"]["z"] = metadata.cloudMetadata.bbCubic.min.z;
    statistics["cloud"]["bbCubic"]["max"]["x"] = metadata.cloudMetadata.bbCubic.max.x;
    statistics["cloud"]["bbCubic"]["max"]["y"] = metadata.cloudMetadata.bbCubic.max.y;
    statistics["cloud"]["bbCubic"]["max"]["z"] = metadata.cloudMetadata.bbCubic.max.z;
    statistics["cloud"]["bbCubic"]["sideLength"] =
            metadata.cloudMetadata.bbCubic.max.x - metadata.cloudMetadata.bbCubic.min.x;
    statistics["cloud"]["offset"]["x"] = metadata.cloudMetadata.cloudOffset.x;
    statistics["cloud"]["offset"]["y"] = metadata.cloudMetadata.cloudOffset.y;
    statistics["cloud"]["offset"]["z"] = metadata.cloudMetadata.cloudOffset.z;
    statistics["cloud"]["scale"]["x"]  = metadata.cloudMetadata.scale.x;
    statistics["cloud"]["scale"]["y"]  = metadata.cloudMetadata.scale.y;
    statistics["cloud"]["scale"]["z"]  = metadata.cloudMetadata.scale.z;

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