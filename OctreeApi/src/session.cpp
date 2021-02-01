//
// Created by KlausP on 01.11.2020.
//

#include <session.h>

#include "json_exporter.h"
#include "sparseOctree.h"
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
    spdlog::debug ("session destroyed");
}

void Session::setPointCloudHost (uint8_t* pointCloud)
{
    itsPointCloud = pointCloud;
    spdlog::debug ("set point cloud data from host");
}

void Session::generateOctree ()
{
    PointCloudMetadata metadata{};
    metadata.cloudType       = itsCloudType;
    metadata.pointAmount     = itsPointAmount;
    metadata.pointDataStride = itsDataStride;
    metadata.scale           = itsScale;
    metadata.cloudOffset     = itsOffset;
    metadata.bbCubic         = itsBoundingBox;

    SparseOctree octree (
            itsChunkingGrid, itsSubsamplingGrid, itsMergingThreshold, metadata, itsSubsamplingStrategy);

    octree.setPointCloudHost (itsPointCloud);

    octree.initialPointCounting ();
    octree.performCellMerging ();
    octree.distributePoints ();

    if (!itsPointDistReportFile.empty ())
    {
        octree.exportHistogram (itsPointDistReportFile, itsPointDistributionBinWidth);
    }

    octree.performSubsampling ();

    if (!itsOctreeExportDirectory.empty ())
    {
        octree.exportPlyNodes (itsOctreeExportDirectory);
    }

    if (!itsJsonReportFile.empty ())
    {
        octree.updateOctreeStatistics ();
        export_json_data (itsJsonReportFile, octree.getMetadata (), octree.getTimings ());
    }

    spdlog::debug ("octree generated");
}

void Session::configureOctreeExport (const string& directory)
{
    itsOctreeExportDirectory = directory;
    spdlog::debug ("Export Octree to: {}", directory);
}

void Session::configureMemoryReport (const std::string& filename)
{
    EventWatcher::getInstance ().configureMemoryReport (filename);
    spdlog::debug ("Export memory report to: {}", filename);
}

void Session::configureJsonReport (const std::string& filename)
{
    itsJsonReportFile = filename;
    spdlog::debug ("Export JSON report to: {}", filename);
}

void Session::configurePointDistributionReport (const std::string& filename, uint32_t binWidth)
{
    itsPointDistReportFile       = filename;
    itsPointDistributionBinWidth = binWidth;
    spdlog::debug ("Export point dist. report to: {}", filename);
}

void Session::configureChunking (uint32_t chunkingGrid, uint32_t mergingThreshold)
{
    itsChunkingGrid     = chunkingGrid;
    itsMergingThreshold = mergingThreshold;
}

void Session::configureSubsampling (uint32_t subsamplingGrid, uint8_t strategy)
{
    itsSubsamplingGrid     = subsamplingGrid;
    itsSubsamplingStrategy = static_cast<SubsamplingStrategy>(strategy);
}
void Session::setCloudType (uint8_t cloudType)
{
    itsCloudType = static_cast<CloudType>(cloudType);
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
