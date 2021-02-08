//
// Created by KlausP on 01.11.2020.
//

#include <session.h>

#include "json_exporter.h"
#include "octree_processor.h"
#include "spdlog/spdlog.h"
#include <iostream>
#include <memory>

Session* Session::ToSession (void* session)
{
    auto s = static_cast<Session*> (session);
    if (s)
    {
        return s;
    }
    throw runtime_error ("No Session is currently initialized!");
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
    itsPointCloud  = pointCloud;
    itsCloudMemory = CLOUD_HOST;
    spdlog::debug ("set point cloud data from host");
}

void Session::generateOctree ()
{
    PointCloudMetadata cloudMetadata{
            itsPointAmount, itsDataStride, itsBoundingBox, itsOffset, itsScale, itsCloudType, itsCloudMemory};

    itsProcessor = std::make_unique<OctreeProcessor> (
            itsPointCloud, itsChunkingGrid, itsMergingThreshold, cloudMetadata, itsSubsamplingMetadata);

    itsProcessor->initialPointCounting ();
    itsProcessor->performCellMerging ();
    itsProcessor->distributePoints ();
    itsProcessor->performSubsampling ();

    spdlog::debug ("octree generated");
}

void Session::exportPotree (const string& directory)
{
    itsProcessor->exportPlyNodes (directory);
    spdlog::debug ("Export Octree to: {}", directory);
}

void Session::exportMemoryReport (const std::string& filename)
{
    EventWatcher::getInstance ().configureMemoryReport (filename);
    spdlog::debug ("Export memory report to: {}", filename);
}

void Session::exportJsonReport (const std::string& filename)
{
    itsProcessor->updateOctreeStatistics ();
    export_json_data (filename, itsProcessor->getMetadata (), itsSubsamplingMetadata, itsProcessor->getTimings ());
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

void Session::configureSubsampling (uint32_t subsamplingGrid, uint8_t strategy, bool averaging, bool replacementScheme)
{
    itsSubsamplingMetadata.performAveraging     = averaging;
    itsSubsamplingMetadata.useReplacementScheme = replacementScheme;
    itsSubsamplingMetadata.subsamplingGrid      = subsamplingGrid;
    itsSubsamplingMetadata.strategy             = static_cast<SubsampleStrategy> (strategy);
}
void Session::setCloudType (uint8_t cloudType)
{
    itsCloudType = static_cast<CloudType> (cloudType);
}

void Session::setCloudBoundingBox (double minX, double minY, double minZ, double maxX, double maxY, double maxZ)
{
    itsBoundingBox = {{minX, minY, minZ}, {maxX, maxY, maxZ}};
}


void Session::setCloudPointAmount (uint32_t pointAmount)
{
    itsPointAmount = pointAmount;
}

void Session::setCloudDataStride (uint32_t dataStride)
{
    itsDataStride = dataStride;
}

void Session::setCloudScale (double x, double y, double z)
{
    itsScale = {x, y, z};
}

void Session::setCloudOffset (double x, double y, double z)
{
    itsOffset = {x, y, z};
}
