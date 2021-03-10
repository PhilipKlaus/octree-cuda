#include "point_cloud.cuh"
#include "time_tracker.cuh"

PointCloudHost::PointCloudHost (uint8_t* source, PointCloudMetadata metadata) : IPointCloud (source, metadata)
{
    auto start     = std::chrono::high_resolution_clock::now ();
    itsDeviceCloud = createGpuU8 (itsMetadata.pointAmount * itsMetadata.pointDataStride, "pointcloud");
    itsDeviceCloud->toGPU (itsSourceCloud);
    auto finish                           = std::chrono::high_resolution_clock::now ();
    std::chrono::duration<double> elapsed = finish - start;
    spdlog::info ("Copy cloud to GPU (incl. memory alloc.) took: {} [s]", elapsed.count ());
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
