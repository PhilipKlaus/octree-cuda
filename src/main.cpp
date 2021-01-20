// Standard library
#include <iostream>
#include <fstream>

// Local dependencies
#include "octreeApi.h"
#include "spdlog/spdlog.h"


using namespace std;

template <typename coordinateType>
void calculateBB(const uint8_t *cloud, PointCloudMetadata &metadata) {

    Vector3<double> minimum {INFINITY, INFINITY, INFINITY};
    Vector3<double> maximum {-INFINITY, -INFINITY, -INFINITY};
    uint8_t byteSize = sizeof (coordinateType);

    for(uint32_t i = 0; i < metadata.pointAmount; ++i)
    {
        minimum.x =
                fmin (minimum.x, (*reinterpret_cast<const coordinateType*> (cloud + i * metadata.pointDataStride)) * metadata.scale.x);
        minimum.y = fmin (
                minimum.y, (*reinterpret_cast<const coordinateType*> (cloud + i * metadata.pointDataStride + byteSize)) * metadata.scale.y);
        minimum.z = fmin (
                minimum.z, (*reinterpret_cast<const coordinateType*> (cloud + i * metadata.pointDataStride + byteSize * 2)) * metadata.scale.z);
        maximum.x =
                fmax (maximum.x, (*reinterpret_cast<const coordinateType*> (cloud + i * metadata.pointDataStride)) * metadata.scale.x);
        maximum.y = fmax (
                maximum.y, (*reinterpret_cast<const coordinateType*> (cloud + i * metadata.pointDataStride + byteSize)) * metadata.scale.y);
        maximum.z = fmax (
                maximum.z, (*reinterpret_cast<const coordinateType*> (cloud + i * metadata.pointDataStride + byteSize * 2)) * metadata.scale.z);
    }

    Vector3<double> dimension{};
    dimension.x = maximum.x - minimum.x;
    dimension.y = maximum.y - minimum.y;
    dimension.z = maximum.z - minimum.z;
    double cubicSideLength = max(max(dimension.x, dimension.y), dimension.z);

    metadata.boundingBox.minimum = {};
    metadata.boundingBox.minimum.x = minimum.x - ((cubicSideLength - dimension.x) / 2.0f);
    metadata.boundingBox.minimum.y = minimum.y - ((cubicSideLength - dimension.y) / 2.0f);
    metadata.boundingBox.minimum.z = minimum.z - ((cubicSideLength - dimension.z) / 2.0f);
    metadata.boundingBox.maximum = {};
    metadata.boundingBox.maximum.x = metadata.boundingBox.minimum.x + cubicSideLength;
    metadata.boundingBox.maximum.y = metadata.boundingBox.minimum.y + cubicSideLength;
    metadata.boundingBox.maximum.z = metadata.boundingBox.minimum.z + cubicSideLength;
    metadata.cloudOffset = metadata.boundingBox.minimum;
    spdlog::info("Original BB: min[x,y,z]=[{},{},{}], max[x,y,z]=[{},{},{}]", minimum.x, minimum.y, minimum.z, maximum.x, maximum.y, maximum.z);
    spdlog::info("Cubic BB: min[x,y,z]=[{},{},{}], max[x,y,z]=[{},{},{}]", metadata.boundingBox.minimum.x, metadata.boundingBox.minimum.y,
                 metadata.boundingBox.minimum.z, metadata.boundingBox.maximum.x, metadata.boundingBox.maximum.y, metadata.boundingBox.maximum.z);
}

int main() {

#ifndef NDEBUG
    spdlog::set_level(spdlog::level::debug);
    ocpi_set_logging_level(0);
#else
    spdlog::set_level(spdlog::level::info);
    ocpi_set_logging_level(1);
#endif

    void *session;
    ocpi_create_session(&session, 0);

    // Setup cloud properties
    PointCloudMetadata metadata = {};
    metadata.pointAmount = 25836417;
    metadata.pointDataStride = 15;
    metadata.scale = {1.f, 1.f, 1.f };
    metadata.cloudType = CloudType::CLOUD_FLOAT_UINT8_T;

    // Read in ply
    ifstream ifs(   "heidentor_color_raw.ply", ios::binary|ios::ate);
    ifstream::pos_type pos = ifs.tellg();
    std::streamoff length = pos;
    auto *pChars = new uint8_t[length];
    ifs.seekg(0, ios::beg);
    ifs.read(reinterpret_cast<char *>(pChars), length);
    ifs.close();

    // Calculate BB
    calculateBB<float>(pChars, metadata);

    // Configurate and create octree
    ocpi_set_point_cloud_metadata(session, metadata);
    ocpi_set_point_cloud_host(session, pChars);
    ocpi_configure_chunking(session, GRID_512, 10000);
    ocpi_configure_subsampling(session, GRID_128, RANDOM_POINT);

    ocpi_configure_point_distribution_report(session, R"(C:\Users\KlausP\Documents\git\master-thesis-klaus\octree_cuda\cmake-build-release\export\histogram.html)", 0);
    ocpi_configure_memory_report(session, R"(C:\Users\KlausP\Documents\git\master-thesis-klaus\octree_cuda\cmake-build-release\export\memory_report.html)");
    ocpi_configure_json_report(session, R"(C:\Users\KlausP\Documents\git\master-thesis-klaus\octree_cuda\cmake-build-release\export\statistics.json)");
    ocpi_configure_octree_export(session,R"(C:\Users\KlausP\Documents\git\master-thesis-klaus\octree_cuda\cmake-build-release\export)");

    ocpi_generate_octree(session);

    ocpi_destroy_session(session);

    delete[] pChars;

}