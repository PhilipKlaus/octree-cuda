//
// Created by KlausP on 04.10.2020.
//

#include "catch2/catch.hpp"
#include "tools.cuh"


TEST_CASE ("Test cuboid sample generation", "[cuboid]") {
    PointCloudMetadata metadata{};
    auto data = tools::generate_point_cloud_cuboid(2, metadata);
    auto host = data->toHost();

    auto* vecData = reinterpret_cast<CoordinateVector<float>*>(host.get());
    // ----------------------
    REQUIRE(vecData[0].x == (0.5));
    REQUIRE(vecData[0].y == (0.5));
    REQUIRE(vecData[0].z == (0.5));
    // ----------------------
    REQUIRE(vecData[1].x == (1.5));
    REQUIRE(vecData[1].y == (0.5));
    REQUIRE(vecData[1].z == (0.5));
    // ----------------------
    REQUIRE(vecData[2].x == (0.5));
    REQUIRE(vecData[2].y == (1.5));
    REQUIRE(vecData[2].z == (0.5));
    // ----------------------
    REQUIRE(vecData[3].x == (1.5));
    REQUIRE(vecData[3].y == (1.5));
    REQUIRE(vecData[3].z == (0.5));
    // ----------------------
    REQUIRE(vecData[4].x == (0.5));
    REQUIRE(vecData[4].y == (0.5));
    REQUIRE(vecData[4].z == (1.5));
    // ----------------------
    REQUIRE(vecData[5].x == (1.5));
    REQUIRE(vecData[5].y == (0.5));
    REQUIRE(vecData[5].z == (1.5));
    // ----------------------
    REQUIRE(vecData[6].x == (0.5));
    REQUIRE(vecData[6].y == (1.5));
    REQUIRE(vecData[6].z == (1.5));
    // ----------------------
    REQUIRE(vecData[7].x == (1.5));
    REQUIRE(vecData[7].y == (1.5));
    REQUIRE(vecData[7].z == (1.5));
}