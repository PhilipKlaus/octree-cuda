#include "../src/point_cloud/point_cloud.cuh"
#include "catch2/catch.hpp"

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
        REQUIRE (data.get () == cloud.getCloudHost ());
    }

    SECTION ("copy the host data to device")
    {
        auto device                       = cloud.getCloudDevice ();
        std::unique_ptr<uint8_t[]> copied = std::make_unique<uint8_t[]> (15 * 1);
        gpuErrchk (cudaMemcpy (copied.get (), device, 15 * 1, cudaMemcpyDeviceToHost));
        REQUIRE (reinterpret_cast<float*> (copied.get ())[0] == 1);
        REQUIRE (reinterpret_cast<float*> (copied.get ())[1] == 2);
        REQUIRE (reinterpret_cast<float*> (copied.get ())[2] == 3);
        REQUIRE (copied[12] == 253);
        REQUIRE (copied[13] == 254);
        REQUIRE (copied[14] == 255);
    }
}