// Standard library
#include <iostream>
#include <fstream>

// Local dependencies
#include "octreeApi.h"
#include "spdlog/spdlog.h"


using namespace std;


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

    PointCloudMetadata metadata = {};
    metadata.pointAmount = 5138448;
    metadata.pointDataStride = 15;
    metadata.scale = {1.f, 1.f, 1.f};

    ifstream ifs("coin_color_headerless.ply", ios::binary|ios::ate);
    ifstream::pos_type pos = ifs.tellg();
    std::streamoff length = pos;
    auto *pChars = new uint8_t[length];
    ifs.seekg(0, ios::beg);
    ifs.read(reinterpret_cast<char *>(pChars), length);
    ifs.close();

    Vector3 minimum {INFINITY, INFINITY, INFINITY};
    Vector3 maximum {-INFINITY, -INFINITY, -INFINITY};

    // Calculate bounding box on CPU
    for(uint32_t i = 0; i < metadata.pointAmount; ++i) {
        minimum.x = fmin(minimum.x, *reinterpret_cast<float*>(pChars + i * metadata.pointDataStride));
        minimum.y = fmin(minimum.y, *reinterpret_cast<float*>(pChars + i * metadata.pointDataStride + 4));
        minimum.z = fmin(minimum.z, *reinterpret_cast<float*>(pChars + i * metadata.pointDataStride + 8));
        maximum.x = fmax(maximum.x, *reinterpret_cast<float*>(pChars + i * metadata.pointDataStride));
        maximum.y = fmax(maximum.y, *reinterpret_cast<float*>(pChars + i * metadata.pointDataStride + 4));
        maximum.z = fmax(maximum.z, *reinterpret_cast<float*>(pChars + i * metadata.pointDataStride + 8));
    }
    spdlog::info(
            "Cloud dimensions: width: {}, height: {}, depth: {}",
            maximum.x-minimum.x, maximum.y-minimum.y, maximum.z-minimum.z);

    metadata.boundingBox.minimum = minimum;
    metadata.boundingBox.maximum = maximum;
    metadata.cloudOffset = minimum;

    ocpi_set_point_cloud_metadata(session, metadata);
    ocpi_load_point_cloud_from_host(session, pChars);
    ocpi_configure_octree(session, 7, 10000);
    ocpi_configure_point_distribution_report(session, R"(C:\Users\KlausP\Documents\git\master-thesis-klaus\octree_cuda\cmake-build-release\export\histogram.html)", 0);

    ocpi_generate_octree(session);
    ocpi_export_ply_nodes(session,R"(C:\Users\KlausP\Documents\git\master-thesis-klaus\octree_cuda\cmake-build-release\export)");
    ocpi_configure_memory_report(session, R"(C:\Users\KlausP\Documents\git\master-thesis-klaus\octree_cuda\cmake-build-release\export\memory_report.html)");

    ocpi_export_time_measurements(session, R"(C:\Users\KlausP\Documents\git\master-thesis-klaus\octree_cuda\cmake-build-release\export\time_measurement.csv)");
    ocpi_export_octree_statistics(session, R"(C:\Users\KlausP\Documents\git\master-thesis-klaus\octree_cuda\cmake-build-release\export\statistics.json)");
    ocpi_destroy_session(session);

    delete[] pChars;

}