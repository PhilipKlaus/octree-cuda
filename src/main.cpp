#include "boundingbox.h"
#include "octreeApi.h"
#include "spdlog/spdlog.h"
#include <fstream>


using namespace std;


std::unique_ptr<uint8_t[]> readPly (const std::string& plyFile)
{
    ifstream ifs (plyFile, ios::binary | ios::ate);
    ifstream::pos_type pos = ifs.tellg ();
    std::streamoff length  = pos;
    auto bytes             = std::make_unique<uint8_t[]> (length);
    ifs.seekg (0, ios::beg);
    ifs.read (reinterpret_cast<char*> (bytes.get ()), length);
    ifs.close ();
    return bytes;
}

int main ()
{
#ifndef NDEBUG
    spdlog::set_level (spdlog::level::debug);
    ocpi_set_logging_level (0);
#else
    spdlog::set_level (spdlog::level::info);
    ocpi_set_logging_level (1);
#endif

    void* session;
    ocpi_create_session (&session, 0);

    // Setup cloud properties
    uint32_t pointAmount     = 47111095;
    uint32_t pointDataStride = 27;
    float scaleX             = 0.001f;
    float scaleY             = 0.001f;
    float scaleZ             = 0.001f;
    auto cloudType           = CloudType::CLOUD_DOUBLE_UINT8_T;
    std::string plyFile      = "lifeboat_headerless.ply";

    // Read in ply
    auto ply = readPly (plyFile);

    // Calculate BB
    auto realBB  = calculateRealBB<double> (ply, pointAmount, pointDataStride);
    auto cubicBB = calculateCubicBB<double> (realBB);

    // Configurate and create octree
    ocpi_set_cloud_type (session, cloudType);
    ocpi_set_cloud_point_amount (session, pointAmount);
    ocpi_set_cloud_data_stride (session, pointDataStride);
    ocpi_set_cloud_scale_f (session, scaleX, scaleY, scaleZ);

    if (cloudType == CLOUD_FLOAT_UINT8_T)
    {
        ocpi_set_cloud_offset_f (session, cubicBB[0], cubicBB[1], cubicBB[2]);
        ocpi_set_cloud_bb_f (session, cubicBB[0], cubicBB[1], cubicBB[2], cubicBB[3], cubicBB[4], cubicBB[5]);
    }
    else
    {
        ocpi_set_cloud_offset_d (session, cubicBB[0], cubicBB[1], cubicBB[2]);
        ocpi_set_cloud_bb_d (session, cubicBB[0], cubicBB[1], cubicBB[2], cubicBB[3], cubicBB[4], cubicBB[5]);
    }

    ocpi_set_point_cloud_host (session, ply.get ());
    ocpi_configure_chunking (session, GRID_512, 10000);
    ocpi_configure_subsampling (session, GRID_128, RANDOM_POINT);

    ocpi_configure_point_distribution_report (session, R"(./export/histogram.html)", 0);
    ocpi_configure_memory_report (session, R"(./export/memory_report.html)");
    ocpi_configure_json_report (session, R"(./statistics.json)");
    ocpi_configure_octree_export (session, R"(./export)");

    ocpi_generate_octree (session);

    ocpi_destroy_session (session);
}