//
// Created by KlausP on 01.11.2020.
//

#include "session.h"

#include <memory>
#include "spdlog/spdlog.h"
#include <iostream>

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

void Session::setMetadata(const PointCloudMetadata &metadata) {
    itsMetadata = metadata;
    spdlog::debug("set metadata");
};

void Session::setPointCloudHost(uint8_t *pointCloud) {
    itsPointCloud = pointCloud;
    spdlog::debug("set point cloud data from host");
}

void Session::generateOctree() {
    if(itsMetadata.cloudType == CloudType::CLOUD_FLOAT_UINT8_T) {
        generateOctreeTemplated<float, uint8_t>();
    }
    else {
        generateOctreeTemplated<double, uint8_t>();
    }
}

template <typename coordinateType, typename colorType>
void Session::generateOctreeTemplated() {

    SparseOctree<coordinateType, colorType> octree(
            itsChunkingGrid,
            itsSubsamplingGrid,
            itsMergingThreshold,
            itsMetadata,
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
        octree.exportOctreeStatistics(itsJsonReportFile);
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
