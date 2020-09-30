#include "catch2/catch.hpp"
#include "../src/tools.cuh"


#include <stdint.h>


TEST_CASE ("Generate equally sampled point cloud", "[equally sampled]") {
    auto data = generate_point_cloud_cuboid(2);
    auto host = data->toHost();

    // ----------------------
    REQUIRE(host[0].x == (0.5));
    REQUIRE(host[0].y == (0.5));
    REQUIRE(host[0].z == (0.5));
    // ----------------------
    REQUIRE(host[1].x == (1.5));
    REQUIRE(host[1].y == (0.5));
    REQUIRE(host[1].z == (0.5));
    // ----------------------
    REQUIRE(host[2].x == (0.5));
    REQUIRE(host[2].y == (1.5));
    REQUIRE(host[2].z == (0.5));
    // ----------------------
    REQUIRE(host[3].x == (1.5));
    REQUIRE(host[3].y == (1.5));
    REQUIRE(host[3].z == (0.5));
    // ----------------------
    REQUIRE(host[4].x == (0.5));
    REQUIRE(host[4].y == (0.5));
    REQUIRE(host[4].z == (1.5));
    // ----------------------
    REQUIRE(host[5].x == (1.5));
    REQUIRE(host[5].y == (0.5));
    REQUIRE(host[5].z == (1.5));
    // ----------------------
    REQUIRE(host[6].x == (0.5));
    REQUIRE(host[6].y == (1.5));
    REQUIRE(host[6].z == (1.5));
    // ----------------------
    REQUIRE(host[7].x == (1.5));
    REQUIRE(host[7].y == (1.5));
    REQUIRE(host[7].z == (1.5));
}
