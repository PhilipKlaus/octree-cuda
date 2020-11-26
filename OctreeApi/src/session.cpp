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
        itsDevice(device),
        itsGlobalOctreeLevel(7),
        itsMergingThreshold(10000)
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

// ToDo: Generalize
void Session::setPointCloudHost(uint8_t *pointCloud) {
    data = make_unique<CudaArray<uint8_t>>(itsMetadata.pointAmount * 12, "pointcloud");
    data->toGPU(pointCloud);
    spdlog::debug("copied point cloud from host to device");
}

void Session::setOctreeProperties(uint16_t globalOctreeLevel, uint32_t mergingThreshold) {
    itsGlobalOctreeLevel = globalOctreeLevel;
    itsMergingThreshold = mergingThreshold;
    itsOctree = make_unique<SparseOctree>(itsGlobalOctreeLevel, itsMergingThreshold, itsMetadata, move(data));

    spdlog::debug("set octree properties");
}

void Session::generateOctree() {
    itsOctree->initialPointCounting();
    itsOctree->performCellMerging();
    itsOctree->distributePoints();
    itsOctree->performSubsampling();
    if(!itsPointDistributionReport.empty()) {
        itsOctree->exportHistogram(itsPointDistributionReport, itsPointDistributionBinWidth);
    }
    spdlog::debug("octree generated");
}

void Session::exportOctree(const string &filename) {
    itsOctree->exportOctree(filename);
    spdlog::debug("octree exported");
}

void Session::configureMemoryReport(const std::string &filename) {
    EventWatcher::getInstance().configureMemoryReport(filename);
    spdlog::debug("configured  memory report: {}", filename);
}

void Session::exportTimeMeasurements(const std::string &filename) {
    itsOctree->exportTimeMeasurements(filename);
    spdlog::debug("exported time measurements");
}

void Session::exportOctreeStatistics(const std::string &filename) {
    itsOctree->exportOctreeStatistics(filename);
    spdlog::debug("exported octree statistics");
}

void Session::configurePointDistributionReport(const std::string &filename, uint32_t binWidth) {
    itsPointDistributionReport = filename;
    itsPointDistributionBinWidth = binWidth;
}
