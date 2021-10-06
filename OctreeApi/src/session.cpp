//
// Created by KlausP on 01.11.2020.
//

#include "spdlog/spdlog.h"
#include <driver_types.h>
#include <iostream>
#include <memory>

#include "defines.cuh"
#include "json_exporter.h"
#include "metadata.cuh"
#include "octree_processor.cuh"
#include "session.h"
#include "time_tracker.cuh"

Session* Session::ToSession (void* session)
{
    auto s = static_cast<Session*> (session);
    if (s)
    {
        return s;
    }
    throw std::runtime_error ("No Session is currently initialized!");
}

Session::Session (int device) : itsDevice (device)
{
    spdlog::debug ("session created");
    setDevice ();
    MemoryTracker::getInstance ().reservedMemoryEvent (0, "Init");
}

void Session::setDevice () const
{
    gpuErrchk (cudaSetDevice (itsDevice));
    cudaDeviceProp props{};
    gpuErrchk (cudaGetDeviceProperties (&props, itsDevice));
    unsigned int flags;
    gpuErrchk (cudaGetDeviceFlags (&flags));
    flags |= cudaDeviceScheduleBlockingSync;
    gpuErrchk (cudaSetDeviceFlags (flags));
    gpuErrchk (cudaGetDeviceFlags (&flags));
    spdlog::info ("Set cudaDeviceScheduleBlockingSync: {}", (flags & cudaDeviceScheduleBlockingSync) > 0 ? "true" : "false");
    spdlog::info ("Using GPU device: {}", props.name);
}

Session::~Session ()
{
    itsProcessor.reset ();
    spdlog::debug ("session destroyed");
}


void Session::setPointCloudHost (uint8_t* pointCloud)
{
    itsPointCloud           = pointCloud;
    itsCloudInfo.memoryType = CLOUD_HOST;
    spdlog::debug ("set point cloud data from host");
}

void Session::generateOctree ()
{
    auto timing = Timing::TimeTracker::start ();
    itsProcessor->initialPointCounting ();
    itsProcessor->performCellMerging ();
    itsProcessor->distributePoints ();
    itsProcessor->performSubsampling ();
    Timing::TimeTracker::stop (timing, "Generating octree", Timing::Time::PROCESS);
}

void Session::exportPotree (const std::string& directory)
{
    auto timing = Timing::TimeTracker::start ();
    itsProcessor->exportPotree (directory);
    Timing::TimeTracker::stop (timing, "Exporting octree", Timing::Time::PROCESS);
}

void Session::exportMemoryReport (const std::string& filename)
{
    MemoryTracker::getInstance ().configureMemoryReport (filename);
    spdlog::debug ("Export memory report to: {}", filename);
}

void Session::exportJsonReport (const std::string& filename)
{
    itsProcessor->updateStatistics ();
    export_json_data (filename, itsProcessingInfo, itsProcessor->getNodeStatistics ());
    spdlog::debug ("Export JSON report to: {}", filename);
}

void Session::exportDistributionHistogram (const std::string& filename, uint32_t binWidth)
{
    itsProcessor->exportHistogram (filename, binWidth);
    spdlog::debug ("Export point dist. report to: {}", filename);
}

void Session::configureChunking (uint32_t chunkingGrid, uint32_t mergingThreshold, float outputFactor)
{
    itsProcessingInfo.chunkingGrid     = chunkingGrid;
    itsProcessingInfo.mergingThreshold = mergingThreshold;
    itsProcessingInfo.outputFactor     = outputFactor;
}

void Session::configureSubsampling (
        uint32_t subsamplingGrid,
        bool intraCellAveraging,
        bool interCellAveraging,
        bool replacementScheme,
        bool useRandomSubsampling)
{
    itsProcessingInfo.useIntraCellAvg      = intraCellAveraging;
    itsProcessingInfo.useInterCellAvg      = interCellAveraging;
    itsProcessingInfo.useReplacementScheme = replacementScheme;
    itsProcessingInfo.subsamplingGrid      = subsamplingGrid;
    itsProcessingInfo.useRandomSubsampling = useRandomSubsampling;
}
void Session::setCloudType (uint8_t cloudType)
{
    itsCloudInfo.cloudType = static_cast<CloudType> (cloudType);
}

void Session::setCloudBoundingBox (double minX, double minY, double minZ, double maxX, double maxY, double maxZ)
{
    itsCloudInfo.bbCubic = {{minX, minY, minZ}, {maxX, maxY, maxZ}};
}


void Session::setCloudPointAmount (uint32_t pointAmount)
{
    itsCloudInfo.pointAmount = pointAmount;
}

void Session::setCloudDataStride (uint32_t dataStride)
{
    itsCloudInfo.pointDataStride = dataStride;
}

void Session::setCloudScale (double x, double y, double z)
{
    itsCloudInfo.scale = {x, y, z};
}

void Session::setCloudOffset (double x, double y, double z)
{
    itsCloudInfo.cloudOffset = {x, y, z};
}

void Session::initOctree ()
{
    itsProcessor = std::make_unique<OctreeProcessor> (itsPointCloud, itsCloudInfo, itsProcessingInfo);
}
