#include "catch2/catch.hpp"
#include "point_cloud.cuh"

TEST_CASE ("A PointCloudHost should")
{
    PointCloudMetadata meta{};
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

    PointCloudHost cloud (data.get (), meta);

    SECTION ("return the original host cloud data")
    {
        CHECK (data.get () == cloud.getCloudHost ());
    }

    SECTION ("copy the host data to device")
    {
        auto device                       = cloud.getCloudDevice ();
        std::unique_ptr<uint8_t[]> copied = std::make_unique<uint8_t[]> (15 * 1);
        REQUIRE (cudaMemcpy (copied.get (), device, 15 * 1, cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK (reinterpret_cast<float*> (copied.get ())[0] == 1);
        CHECK (reinterpret_cast<float*> (copied.get ())[1] == 2);
        CHECK (reinterpret_cast<float*> (copied.get ())[2] == 3);
        CHECK (copied[12] == 253);
        CHECK (copied[13] == 254);
        CHECK (copied[14] == 255);
    }

    SECTION ("create and copy the output buffer to host") {
        REQUIRE(cloud.getOutputBuffer_d() != nullptr);
        REQUIRE(cloud.getOutputBufferSize() == sizeof (OutputBuffer) * 2);
        REQUIRE(cudaMemset (cloud.getOutputBuffer_d(), 1, sizeof (OutputBuffer) * 2) == cudaSuccess);

        std::unique_ptr<OutputBuffer[]> out = cloud.getOutputBuffer_h();
        REQUIRE(out != nullptr);
        CHECK(out[0].x == static_cast<uint32_t>(0x1010101));
        CHECK(out[0].y == static_cast<uint32_t>(0x1010101));
        CHECK(out[0].z == static_cast<uint32_t>(0x1010101));
        CHECK(out[0].r == static_cast<uint16_t>(0x101));
        CHECK(out[0].g == static_cast<uint16_t>(0x101));
        CHECK(out[0].b == static_cast<uint16_t>(0x101));
    }
}