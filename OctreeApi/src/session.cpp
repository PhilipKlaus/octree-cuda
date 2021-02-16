//
// Created by KlausP on 01.11.2020.
//

#include "spdlog/spdlog.h"
#include <driver_types.h>
#include <iostream>
#include <memory>

#include "defines.cuh"
#include "json_exporter.h"
#include "metadata.h"
#include "octree_processor.h"
#include "session.h"

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
    EventWatcher::getInstance ().reservedMemoryEvent (0, "Session created");
}

void Session::setDevice () const
{
    gpuErrchk (cudaSetDevice (itsDevice));
    cudaDeviceProp props{};
    gpuErrchk (cudaGetDeviceProperties (&props, itsDevice));
    spdlog::info ("Using GPU device: {}", props.name);
}

Session::~Session ()
{
    itsProcessor.reset ();
    spdlog::debug ("session destroyed");
}


void Session::setPointCloudHost (uint8_t* pointCloud)
{
    itsPointCloud               = pointCloud;
    itsCloudMetadata.memoryType = CLOUD_HOST;
    spdlog::debug ("set point cloud data from host");
}

void Session::generateOctree ()
{
    itsProcessor = std::make_unique<OctreeProcessor> (
            itsPointCloud, itsChunkingGrid, itsMergingThreshold, itsCloudMetadata, itsSubsamplingMetadata);

    itsProcessor->initialPointCounting ();
    itsProcessor->performCellMerging ();
    itsProcessor->distributePoints ();
    itsProcessor->performSubsampling ();

    spdlog::debug ("octree generated");
}

void Session::exportPotree (const std::string& directory)
{
    itsProcessor->exportPotree (directory);
    spdlog::debug ("Export Octree to: {}", directory);
}

void Session::exportMemoryReport (const std::string& filename)
{
    EventWatcher::getInstance ().configureMemoryReport (filename);
    spdlog::debug ("Export memory report to: {}", filename);
}

void Session::exportJsonReport (const std::string& filename)
{
    itsProcessor->updateStatistics ();
    export_json_data (
            filename,
            itsProcessor->getOctreeMetadata (),
            itsCloudMetadata,
            itsSubsamplingMetadata,
            itsProcessor->getTimings ());
    spdlog::debug ("Export JSON report to: {}", filename);
}

void Session::exportDistributionHistogram (const std::string& filename, uint32_t binWidth)
{
    itsProcessor->exportHistogram (filename, binWidth);
    spdlog::debug ("Export point dist. report to: {}", filename);
}

void Session::configureChunking (uint32_t chunkingGrid, uint32_t mergingThreshold)
{
    itsChunkingGrid     = chunkingGrid;
    itsMergingThreshold = mergingThreshold;
}

void Session::configureSubsampling (uint32_t subsamplingGrid, bool averaging, bool replacementScheme)
{
    itsSubsamplingMetadata.performAveraging     = averaging;
    itsSubsamplingMetadata.useReplacementScheme = replacementScheme;
    itsSubsamplingMetadata.subsamplingGrid      = subsamplingGrid;
}
void Session::setCloudType (uint8_t cloudType)
{
    itsCloudMetadata.cloudType = static_cast<CloudType> (cloudType);
}

void Session::setCloudBoundingBox (double minX, double minY, double minZ, double maxX, double maxY, double maxZ)
{
    itsCloudMetadata.bbCubic = {{minX, minY, minZ}, {maxX, maxY, maxZ}};
}


void Session::setCloudPointAmount (uint32_t pointAmount)
{
    itsCloudMetadata.pointAmount = pointAmount;
}

void Session::setCloudDataStride (uint32_t dataStride)
{
    itsCloudMetadata.pointDataStride = dataStride;
}

void Session::setCloudScale (double x, double y, double z)
{
    itsCloudMetadata.scale = {x, y, z};
}

void Session::setCloudOffset (double x, double y, double z)
{
    itsCloudMetadata.cloudOffset = {x, y, z};
}
