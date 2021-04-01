//
// Created by KlausP on 25.01.2021.
//

#pragma once

#include "memory_tracker.cuh"
#include "metadata.cuh"
#include "time_tracker.cuh"
#include <iomanip>
#include <json.hpp>


void export_json_data (
        const std::string& filePath, const ProcessingInfo& subsampleInfo, const OctreeInfo& nodeStatistics)
{
    nlohmann::ordered_json statistics;
    statistics["depth"] = nodeStatistics.depth;

    statistics["chunking"]["grid"]             = subsampleInfo.chunkingGrid;
    statistics["chunking"]["mergingThreshold"] = subsampleInfo.mergingThreshold;

    statistics["subsampling"]["grid"]     = subsampleInfo.subsamplingGrid;
    statistics["subsampling"]["strategy"] = "RANDOM POINT";

    statistics["resultNodes"]["octreeNodes"]      = nodeStatistics.leafNodeAmount + nodeStatistics.parentNodeAmount;
    statistics["resultNodes"]["leafNodeAmount"]   = nodeStatistics.leafNodeAmount;
    statistics["resultNodes"]["parentNodeAmount"] = nodeStatistics.parentNodeAmount;
    statistics["resultNodes"]["maxLeafDepth"]     = nodeStatistics.maxLeafDepth;

    statistics["overallNodes"]["sparseOctreeNodes"] = nodeStatistics.nodeAmountSparse;
    statistics["overallNodes"]["denseOctreeNodes"]  = nodeStatistics.nodeAmountDense;
    statistics["overallNodes"]["memorySaving"] =
            (1 - (static_cast<float> (nodeStatistics.nodeAmountSparse) / nodeStatistics.nodeAmountDense)) * 100;

    statistics["pointDistribution"]["meanPointsPerLeafNode"]  = nodeStatistics.meanPointsPerLeafNode;
    statistics["pointDistribution"]["stdevPointsPerLeafNode"] = nodeStatistics.stdevPointsPerLeafNode;
    statistics["pointDistribution"]["minPointsPerNode"]       = nodeStatistics.minPointsPerNode;
    statistics["pointDistribution"]["maxPointsPerNode"]       = nodeStatistics.maxPointsPerNode;

    auto& tracker = Timing::TimeTracker::getInstance ();

    float accumulatedKernelTime = 0;
    for (auto const& gpuTime : tracker.getKernelTimings ())
    {
        statistics["timings"]["kernel"][std::get<0> (gpuTime)]["invocations"] = std::get<1> (gpuTime).invocations;
        statistics["timings"]["kernel"][std::get<0> (gpuTime)]["duration"]    = std::get<1> (gpuTime).duration;

        accumulatedKernelTime += std::get<1> (gpuTime).duration;
    }

    double accumulatedMemCpyTime = 0;
    for (auto const& memCpyTime : tracker.getMemCpyTimings ())
    {
        statistics["timings"]["memcpy"][std::get<1> (memCpyTime)] = std::get<0> (memCpyTime);
        accumulatedMemCpyTime += std::get<0> (memCpyTime);
    }

    double accumulatedMallocTime = 0;
    for (auto const& memAllocTime : tracker.getMemAllocTimings ())
    {
        statistics["timings"]["cudamalloc"][std::get<1> (memAllocTime)] = std::get<0> (memAllocTime);
        accumulatedMallocTime += std::get<0> (memAllocTime);
    }

    double accumulatedProcessTimings = 0;
    for (auto const& memAllocTime : tracker.getProcessTimings ())
    {
        statistics["timings"]["process"][std::get<1> (memAllocTime)] = std::get<0> (memAllocTime);
        accumulatedProcessTimings += std::get<0> (memAllocTime);
    }

    statistics["timings"]["accumulatedKernelTime"]     = accumulatedKernelTime;
    statistics["timings"]["accumulatedMemCpyTime"]     = accumulatedMemCpyTime;
    statistics["timings"]["accumulatedMallocTime"]     = accumulatedMallocTime;
    statistics["timings"]["accumulatedProcessTimings"] = accumulatedProcessTimings;

    MemoryTracker& watcher                    = MemoryTracker::getInstance ();
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