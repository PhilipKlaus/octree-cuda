#include "catch2/catch.hpp"
#include "../src/tools.cuh"
#include "../src/chunking.cuh"


TEST_CASE ("Test generation of equally sampled point cloud cuboid", "[cuboid]") {
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

TEST_CASE ("Test initial point counting ", "[counting]") {

    // Create test data point cloud
    auto pointCloud = generate_point_cloud_cuboid(128);

    Vector3 posOffset = {0.5, 0.5 , 0.5};
    Vector3 size = {127, 127, 127};
    Vector3 minimum = {0.5, 0.5, 0.5};

    // Perform initial point counting -> octree level 8
    auto countingGrid = initialPointCounting(move(pointCloud), 128, posOffset, size, minimum);

    // Test if each point fall exactly in one cell
    auto host = countingGrid->toHost();
    int max = static_cast<int>(pow(128, 3));
    for(int i = 0; i < pow(128, 3); ++i) {
        REQUIRE(host[i] == 1);
    }
}
