#include "point_cloud.cuh"
#include "time_tracker.cuh"

PointCloudHost::PointCloudHost (uint8_t* source, PointCloudMetadata metadata) : IPointCloud (source, metadata)
{
    auto timing    = Timing::TimeTracker::start ();
    itsDeviceCloud = createGpuU8 (itsMetadata.pointAmount * itsMetadata.pointDataStride, "pointCloud");
    itsDeviceCloud->toGPU (itsSourceCloud);
    Timing::TimeTracker::stop (timing, "Copy point cloud to GPU", Timing::Time::PROCESS);
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
    }
    return itsHostCloud.get ();
}

uint8_t* PointCloudDevice::getCloudDevice ()
{
    return itsSourceCloud;
}
