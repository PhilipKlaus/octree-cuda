//
// Created by KlausP on 03.02.2021.
//

#pragma once


#include "metadata.h"
#include <cstdint>
#include <types.cuh>

class IPointCloud
{
public:
    IPointCloud (uint8_t* source, PointCloudMetadata metadata) : itsSourceCloud (source), itsMetadata (metadata)
    {}
    virtual ~IPointCloud ()            = default;
    virtual uint8_t* getCloudHost ()   = 0;
    virtual uint8_t* getCloudDevice () = 0;

    const PointCloudMetadata& getMetadata ()
    {
        return itsMetadata;
    }

protected:
    uint8_t* itsSourceCloud;
    PointCloudMetadata itsMetadata;
};

class PointCloudHost : public IPointCloud
{
public:
    PointCloudHost (uint8_t* source, PointCloudMetadata metadata);
    uint8_t* getCloudHost () override;
    uint8_t* getCloudDevice () override;

private:
    GpuArrayU8 itsDeviceCloud;
};

class PointCloudDevice : public IPointCloud
{
public:
    PointCloudDevice (uint8_t* source, PointCloudMetadata metadata);
    uint8_t* getCloudHost () override;
    uint8_t* getCloudDevice () override;

private:
    std::unique_ptr<uint8_t[]> itsHostCloud;
};

using PointCloud = std::unique_ptr<IPointCloud>;
