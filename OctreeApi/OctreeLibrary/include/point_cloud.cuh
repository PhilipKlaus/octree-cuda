//
// Created by KlausP on 03.02.2021.
//

#pragma once

#include "api_types.h"
#include <cstdint>
#include <types.cuh>

class PointCloud
{
public:
    PointCloud (uint8_t* source, PointCloudMetadata metadata) :
            itsSourceCloud (source), itsMetadata (metadata)
    {}
    virtual uint8_t* getCloudHost ()   = 0;
    virtual uint8_t* getCloudDevice () = 0;

protected:
    uint8_t* itsSourceCloud;
    PointCloudMetadata itsMetadata;
};

class PointCloudHost : public PointCloud
{
public:
    PointCloudHost (uint8_t* source, PointCloudMetadata metadata);
    uint8_t* getCloudHost () override;
    uint8_t* getCloudDevice () override;

private:
    GpuArrayU8 itsDeviceCloud;
};

class PointCloudDevice : public PointCloud
{
public:
    PointCloudDevice (uint8_t* source, PointCloudMetadata metadata);
    uint8_t* getCloudHost () override;
    uint8_t* getCloudDevice () override;

private:
    std::unique_ptr<uint8_t> itsHostCloud;
};
