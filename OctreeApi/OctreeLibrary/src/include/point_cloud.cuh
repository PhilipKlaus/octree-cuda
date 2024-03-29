/**
 * @file point_cloud.cuh
 * @author Philip Klaus
 * @brief Contains code for wrapping point cloud data.
 */

#pragma once


#include "metadata.cuh"
#include "types.cuh"
#include <cstdint>

/**
 * An interface for a generic point cloud.
 * It provides methods for accessing point cloud data residing on host and device.
 */
class IPointCloud
{
public:
    IPointCloud (uint8_t* source, PointCloudInfo metadata, float outputFactor) :
            itsSourceCloud (source), itsMetadata (metadata)
    {
        auto expectedPoints = static_cast<uint64_t> (itsMetadata.pointAmount * outputFactor);
        auto timing         = Timing::TimeTracker::start ();
        itsOutput           = createGpuOutputBuffer (expectedPoints, "outputBuffer");
        Timing::TimeTracker::stop (timing, "Create output buffer", Timing::Time::PROCESS);
    }
    virtual ~IPointCloud ()            = default;
    virtual uint8_t* getCloudHost ()   = 0;
    virtual uint8_t* getCloudDevice () = 0;

    const PointCloudInfo& getMetadata ()
    {
        return itsMetadata;
    }

    OutputBuffer* getOutputBuffer_d ()
    {
        return itsOutput->devicePointer ();
    }

    std::unique_ptr<OutputBuffer[]> getOutputBuffer_h ()
    {
        return itsOutput->toHost ();
    }

    uint64_t getOutputBufferSize ()
    {
        return itsOutput->pointCount () * sizeof (OutputBuffer);
    }

protected:
    uint8_t* itsSourceCloud;
    GpuOutputBuffer itsOutput;
    PointCloudInfo itsMetadata;
};

/**
 * A point cloud which memory resides on the host-side.
 */
class PointCloudHost : public IPointCloud
{
public:
    PointCloudHost (uint8_t* source, PointCloudInfo metadata, float outputFactor);
    uint8_t* getCloudHost () override;
    uint8_t* getCloudDevice () override;

private:
    GpuArrayU8 itsDeviceCloud;
};

/**
 * A point cloud which memory resides on the device-side.
 */
class PointCloudDevice : public IPointCloud
{
public:
    PointCloudDevice (uint8_t* source, PointCloudInfo metadata, float outputFactor);
    uint8_t* getCloudHost () override;
    uint8_t* getCloudDevice () override;

private:
    std::unique_ptr<uint8_t[]> itsHostCloud;
};

using PointCloud = std::unique_ptr<IPointCloud>;
