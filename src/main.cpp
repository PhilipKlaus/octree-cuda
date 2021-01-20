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
    metadata.pointAmount = 119701547;
    metadata.pointDataStride = 27;
    //metadata.scale = {pow(10,-250), pow(10,-250), pow(10,-250)};
    metadata.scale = {1.f, 1.f, 1.f };

    ifstream ifs("morrobay_fused_headerless.ply", ios::binary|ios::ate);
    ifstream::pos_type pos = ifs.tellg();
    std::streamoff length = pos;
    auto *pChars = new uint8_t[length];
    ifs.seekg(0, ios::beg);
    ifs.read(reinterpret_cast<char *>(pChars), length);
    ifs.close();

    Vector3<double> minimum {INFINITY, INFINITY, INFINITY};
    Vector3<double> maximum {-INFINITY, -INFINITY, -INFINITY};

    // Calculate bounding box on CPU
    for(uint32_t i = 0; i < metadata.pointAmount; ++i) {
        minimum.x = fmin(minimum.x, (*reinterpret_cast<double*>(pChars + i * metadata.pointDataStride)) * metadata.scale.x);
        minimum.y = fmin(minimum.y, (*reinterpret_cast<double*>(pChars + i * metadata.pointDataStride + 8)) * metadata.scale.y);
        minimum.z = fmin(minimum.z, (*reinterpret_cast<double*>(pChars + i * metadata.pointDataStride + 16)) * metadata.scale.z);
        maximum.x = fmax(maximum.x, (*reinterpret_cast<double*>(pChars + i * metadata.pointDataStride)) * metadata.scale.x);
        maximum.y = fmax(maximum.y, (*reinterpret_cast<double*>(pChars + i * metadata.pointDataStride + 8)) * metadata.scale.y);
        maximum.z = fmax(maximum.z, (*reinterpret_cast<double*>(pChars + i * metadata.pointDataStride + 16)) * metadata.scale.z);
    }

    Vector3<double> dimension{};
    dimension.x = maximum.x - minimum.x;
    dimension.y = maximum.y - minimum.y;
    dimension.z = maximum.z - minimum.z;
    double cubicSideLength = max(max(dimension.x, dimension.y), dimension.z);
    spdlog::info("Cloud dimensions: width: {}, height: {}, depth: {}", dimension.x, dimension.y, dimension.z);
    spdlog::info("Cubic BB sidelength: {}", cubicSideLength);

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

    ocpi_set_point_cloud_metadata(session, metadata);
    ocpi_load_point_cloud_from_host(session, pChars);
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