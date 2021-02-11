#include "point_cloud.cuh"

PointCloudHost::PointCloudHost (uint8_t* source, PointCloudMetadata metadata) : IPointCloud (source, metadata)
{
    itsDeviceCloud = createGpuU8 (itsMetadata.pointAmount * itsMetadata.pointDataStride, "pointcloud");
    itsDeviceCloud->toGPU (itsSourceCloud);
    spdlog::info ("Copied point cloud from host->device");
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
        spdlog::info ("Copied point cloud from device->host");
    }
    return itsHostCloud.get ();
}
uint8_t* PointCloudDevice::getCloudDevice ()
{
    return itsSourceCloud;
}
