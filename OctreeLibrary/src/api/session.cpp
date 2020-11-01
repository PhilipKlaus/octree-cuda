//
// Created by KlausP on 01.11.2020.
//

#include "session.h"

#include <memory>
#include "spdlog/spdlog.h"
#include "../src/defines.cuh"

#include <iostream>
Session* Session::ToSession (void* session)
{
    auto s = static_cast<Session*> (session);
    if (s)
    {
        return s;
    }
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

const PointCloudMetadata& Session::getMetadata() const {
    return itsMetadata;
}

void Session::setPointCloudHost(uint8_t *pointCloud) {
    data = make_unique<CudaArray<Vector3>>(itsMetadata.pointAmount, "pointcloud");
    data->toGPU(pointCloud);
    itsPointCloud = make_unique<PointCloud>(move(data));
    spdlog::debug("copied point cloud from host to device");
}

void Session::setOctreeProperties(uint16_t globalOctreeLevel, uint32_t mergingThreshold) {
    itsGlobalOctreeLevel = globalOctreeLevel;
    itsMergingThreshold = mergingThreshold;
    spdlog::debug("set octree properties");
}

void Session::generateOctree() {
    itsPointCloud->getMetadata() = itsMetadata;
    itsPointCloud->initialPointCountingSparse(itsGlobalOctreeLevel);
    itsPointCloud->performCellMergingSparse(itsMergingThreshold);
    itsPointCloud->distributePointsSparse();
}

void Session::exportOctree(Vector3 *cpuPointCloud) {
    itsPointCloud->exportOctreeSparse(cpuPointCloud);
}

void Session::configureMemoryReport(const std::string &filename) {
    EventWatcher::getInstance().configureMemoryReport(filename);
}

