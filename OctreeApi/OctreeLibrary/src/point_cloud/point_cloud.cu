#include "point_cloud.cuh"
#include "time_tracker.cuh"

PointCloudHost::PointCloudHost (uint8_t* source, PointCloudMetadata metadata) : IPointCloud (source, metadata)
{
    itsDeviceCloud = createGpuU8 (itsMetadata.pointAmount * itsMetadata.pointDataStride, "pointcloud");

    auto start = std::chrono::high_resolution_clock::now ();
    itsDeviceCloud->toGPU (itsSourceCloud);
    auto stop                             = std::chrono::high_resolution_clock::now ();
    std::chrono::duration<double> elapsed = stop - start;
    TimeTracker::getInstance ().trackMemCpyTime (elapsed.count () * 1000, "point cloud", true);
}
uint8_t* PointCloudHost::getCloudHost ()
{
    return itsSourceCloud;
}
uint8_t* PointCloudHost::getCloudDevice ()
{
    return itsDeviceCloud->devicePointer ();
}


PointCloudDevice::PointCloudDevice (uint8_t* source, PointCloudMetadata metadata) : IPointCloud (source, metadata)
{}

uint8_t* PointCloudDevice::getCloudHost ()
{
    if (!itsHostCloud)
    {
        uint64_t cloudByteSize = itsMetadata.pointAmount * itsMetadata.pointDataStride;
        itsHostCloud           = std::make_unique<uint8_t[]> (cloudByteSize);
        gpuErrchk (cudaMemcpy (
                itsHostCloud.get (), itsSourceCloud, sizeof (uint8_t) * cloudByteSize, cudaMemcpyDeviceToHost));
        spdlog::info ("[memcpy] Copied point cloud from device->host");
    }
    return itsHostCloud.get ();
}
uint8_t* PointCloudDevice::getCloudDevice ()
{
    return itsSourceCloud;
}
