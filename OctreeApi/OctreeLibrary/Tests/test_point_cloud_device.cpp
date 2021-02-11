#include "catch2/catch.hpp"
#include "point_cloud.cuh"

TEST_CASE ("A PointCloudDevice should")
{
    PointCloudMetadata meta{};
    meta.pointAmount     = 1;
    meta.pointDataStride = 15;
    meta.bbCubic.min     = {-1, -1, -1};
    meta.bbCubic.max     = {1, 1, 1};
    meta.cloudOffset     = {0, 0, 0};
    meta.scale           = {1, 1, 1};
    meta.cloudType = CLOUD_FLOAT_UINT8_T, meta.memoryType = CLOUD_HOST;

    std::unique_ptr<uint8_t []> data = std::make_unique<uint8_t []>(15 * 1);
    reinterpret_cast<float*>(data.get())[0] = 1;
    reinterpret_cast<float*>(data.get())[1] = 2;
    reinterpret_cast<float*>(data.get())[2] = 3;
    data[12] = 253;
    data[13] = 254;
    data[14] = 255;

    uint8_t *device = nullptr;
    gpuErrchk (cudaMalloc ((void**)&device, 15 * 1));
    gpuErrchk (cudaMemcpy (device, data.get(), 15 * 1, cudaMemcpyHostToDevice));
    PointCloudDevice cloud (device, meta);

    SECTION("return the original device cloud data") {
        REQUIRE(device == cloud.getCloudDevice());
    }

    SECTION("copy the device data to host") {
        auto host = cloud.getCloudHost();
        REQUIRE(reinterpret_cast<float*>(host)[0] == 1);
        REQUIRE(reinterpret_cast<float*>(host)[1] == 2);
        REQUIRE(reinterpret_cast<float*>(host)[2] == 3);
        REQUIRE(host[12] == 253);
        REQUIRE(host[13] == 254);
        REQUIRE(host[14] == 255);
    }
}