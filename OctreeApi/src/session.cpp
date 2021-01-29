//
// Created by KlausP on 01.11.2020.
//

#include <session.h>

#include <memory>
#include <iostream>
#include "spdlog/spdlog.h"
#include "sparseOctree.h"
#include "json_exporter.h"

Session* Session::ToSession (void* session)
{
    auto s = static_cast<Session*> (session);
    if (s)
    {
        return s;
    }
    throw runtime_error("No Session is currently initialized!");
}

Session::Session(int device):
        itsDevice(device)
{
    spdlog::debug("session created");
    setDevice ();
    EventWatcher::getInstance().reservedMemoryEvent(0, "Session created");
}

void Session::setDevice() const {
    gpuErrchk (cudaSetDevice (itsDevice));
    cudaDeviceProp props{};
    gpuErrchk(cudaGetDeviceProperties(&props, itsDevice));
    spdlog::info("Using GPU device: {}", props.name);
}

Session::~Session() {
    spdlog::debug("session destroyed");
}

void Session::setPointCloudHost(uint8_t *pointCloud) {
    itsPointCloud = pointCloud;
    spdlog::debug("set point cloud data from host");
}

void Session::generateOctree() {
    if(itsCloudType == CloudType::CLOUD_FLOAT_UINT8_T) {
        PointCloudMetadata<float> metadata{};
        metadata.cloudType = itsCloudType;
        metadata.pointAmount = itsPointAmount;
        metadata.pointDataStride = itsDataStride;
        metadata.scale = itsScaleF;
        metadata.cloudOffset = itsOffsetF;
        metadata.bbCubic         = itsBoundingBoxF;
        generateOctreeTemplated<float, uint8_t>(metadata);
    }
    else {
        PointCloudMetadata<double> metadata{};
        metadata.cloudType = itsCloudType;
        metadata.pointAmount = itsPointAmount;
        metadata.pointDataStride = itsDataStride;
        metadata.scale = itsScaleD;
        metadata.cloudOffset = itsOffsetD;
        metadata.bbCubic         = itsBoundingBoxD;
        generateOctreeTemplated<double, uint8_t>(metadata);
    }
}

template <typename coordinateType, typename colorType>
void Session::generateOctreeTemplated(PointCloudMetadata<coordinateType> metadata) {

    SparseOctree<coordinateType, colorType> octree(
            itsChunkingGrid,
            itsSubsamplingGrid,
            itsMergingThreshold,
            metadata,
            itsSubsamplingStrategy
            );

    octree.setPointCloudHost(itsPointCloud);

    octree.initialPointCounting();
    octree.performCellMerging();
    octree.distributePoints();

    if(!itsPointDistReportFile.empty()) {
        octree.exportHistogram(itsPointDistReportFile, itsPointDistributionBinWidth);
    }

    octree.performSubsampling();

    if(!itsOctreeExportDirectory.empty()) {
        octree.exportPlyNodes(itsOctreeExportDirectory);
    }

    if(!itsJsonReportFile.empty()) {
        octree.updateOctreeStatistics();
        export_json_data(itsJsonReportFile, octree.getMetadata(), octree.getTimings());
    }

    spdlog::debug("octree generated");
}

void Session::configureOctreeExport(const string &directory) {
    itsOctreeExportDirectory = directory;
    spdlog::debug("Export Octree to: {}", directory);
}

void Session::configureMemoryReport(const std::string &filename) {
    EventWatcher::getInstance().configureMemoryReport(filename);
    spdlog::debug("Export memory report to: {}", filename);
}

void Session::configureJsonReport(const std::string &filename) {
    itsJsonReportFile = filename;
    spdlog::debug("Export JSON report to: {}", filename);
}

void Session::configurePointDistributionReport(const std::string &filename, uint32_t binWidth) {
    itsPointDistReportFile = filename;
    itsPointDistributionBinWidth = binWidth;
    spdlog::debug("Export point dist. report to: {}", filename);
}

void Session::configureChunking(GridSize chunkingGrid, uint32_t mergingThreshold) {
    itsChunkingGrid = chunkingGrid;
    itsMergingThreshold = mergingThreshold;
}

void Session::configureSubsampling(GridSize subsamplingGrid, SubsamplingStrategy strategy) {
    itsSubsamplingGrid = subsamplingGrid;
    itsSubsamplingStrategy = strategy;
}
void Session::setCloudType (CloudType cloudType)
{
    itsCloudType = cloudType;
}

void Session::setCloudBoundingBoxF (
        float minX, float minY, float minZ, float maxX, float maxY, float maxZ)
{
    itsBoundingBoxF = {
            {minX, minY, minZ},
            {maxX, maxY, maxZ}
    };
}

void Session::setCloudBoundingBoxD (
        double minX, double minY, double minZ, double maxX, double maxY, double maxZ)
{
    itsBoundingBoxD = {
            {minX, minY, minZ},
            {maxX, maxY, maxZ}
    };
}
void Session::setCloudPointAmount (uint32_t pointAmount)
{
    itsPointAmount = pointAmount;
}

void Session::setCloudDataStride (uint32_t dataStride)
{
    itsDataStride = dataStride;
}

void Session::setCloudScaleF (float x, float y, float z)
{
    itsScaleF = { x, y, z};
}

void Session::setCloudOffsetF (float x, float y, float z)
{
    itsOffsetF = { x, y, z};
}

void Session::setCloudScaleD (double x, double y, double z)
{
    itsScaleD = { x, y, z};
}

void Session::setCloudOffsetD (double x, double y, double z)
{
    itsOffsetD = { x, y, z};
}
