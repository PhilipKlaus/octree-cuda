#include "catch2/catch.hpp"
#include "point_cloud.cuh"

TEST_CASE ("A PointCloudDevice should")
{
    PointCloudInfo meta{};
    meta.pointAmount     = 1;
    meta.pointDataStride = 15;
    meta.bbCubic.min     = {-1, -1, -1};
    meta.bbCubic.max     = {1, 1, 1};
    meta.cloudOffset     = {0, 0, 0};
    meta.scale           = {1, 1, 1};
    meta.cloudType = CLOUD_FLOAT_UINT8_T, meta.memoryType = CLOUD_HOST;

    std::unique_ptr<uint8_t[]> data           = std::make_unique<uint8_t[]> (15 * 1);
    reinterpret_cast<float*> (data.get ())[0] = 1;
    reinterpret_cast<float*> (data.get ())[1] = 2;
    reinterpret_cast<float*> (data.get ())[2] = 3;
    data[12]                                  = 253;
    data[13]                                  = 254;
    data[14]                                  = 255;

    uint8_t* device = nullptr;
    gpuErrchk (cudaMalloc ((void**)&device, 15 * 1));
    gpuErrchk (cudaMemcpy (device, data.get (), 15 * 1, cudaMemcpyHostToDevice));
    PointCloudDevice cloud (device, meta, 2.2f);

    SECTION ("return the original device cloud data")
    {
        REQUIRE (device == cloud.getCloudDevice ());
    }

    SECTION ("copy the device data to host")
    {
        auto host = cloud.getCloudHost ();
        REQUIRE (reinterpret_cast<float*> (host)[0] == 1);
        REQUIRE (reinterpret_cast<float*> (host)[1] == 2);
        REQUIRE (reinterpret_cast<float*> (host)[2] == 3);
        REQUIRE (host[12] == 253);
        REQUIRE (host[13] == 254);
        REQUIRE (host[14] == 255);
    }

    SECTION ("create and copy the output buffer to host")
    {
        std::unique_ptr<OutputBuffer[]> out = cloud.getOutputBuffer_h ();
        REQUIRE (out != nullptr);
        CHECK (out[0].x == static_cast<uint32_t> (0x0000000));
        CHECK (out[0].y == static_cast<uint32_t> (0x0000000));
        CHECK (out[0].z == static_cast<uint32_t> (0x0000000));
        CHECK (out[0].r == static_cast<uint16_t> (0x000));
        CHECK (out[0].g == static_cast<uint16_t> (0x000));
        CHECK (out[0].b == static_cast<uint16_t> (0x000));

        REQUIRE (cloud.getOutputBuffer_d () != nullptr);
        REQUIRE (cloud.getOutputBufferSize () == sizeof (OutputBuffer) * 2);
        REQUIRE (cudaMemset (cloud.getOutputBuffer_d (), 1, sizeof (OutputBuffer) * 2) == cudaSuccess);

        out = cloud.getOutputBuffer_h ();
        REQUIRE (out != nullptr);
        CHECK (out[0].x == static_cast<uint32_t> (0x1010101));
        CHECK (out[0].y == static_cast<uint32_t> (0x1010101));
        CHECK (out[0].z == static_cast<uint32_t> (0x1010101));
        CHECK (out[0].r == static_cast<uint16_t> (0x101));
        CHECK (out[0].g == static_cast<uint16_t> (0x101));
        CHECK (out[0].b == static_cast<uint16_t> (0x101));
    }
}