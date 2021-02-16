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

    bool isAveraging = true;
    bool useReplacementScheme = true;
    uint32_t chunkingGrid = 512;
    uint32_t subsamplingGrid = 128;
    uint32_t mergingThreshold = 10000;

    // Setup cloud properties
    /*uint32_t pointAmount     = 25010001;
    uint32_t pointDataStride = 15;
    float scaleX             = 0.001f;
    float scaleY             = 0.001f;
    float scaleZ             = 0.001f;
    auto cloudType           = 0;
    std::string plyFile      = "wave_headerless.ply";
    */

 /*        uint32_t pointAmount     = 1344573;
    uint32_t pointDataStride = 27;
    float scaleX             = 0.01f;
    float scaleY             = 0.01f;
    float scaleZ             = 0.01f;
    auto cloudType           = 1;
    std::string plyFile      = "testsmall_headerless.ply";
*/
/*uint32_t pointAmount     = 5138448;
uint32_t pointDataStride = 43;
float scaleX             = 0.001f;
float scaleY             = 0.001f;
float scaleZ             = 0.001f;
auto cloudType           = 0;
std::string plyFile      = "coin_2320x9x2x4000_headerless.ply";
*/

uint32_t pointAmount     = 25836417;
uint32_t pointDataStride = 15;
float scaleX             = 0.001f;
float scaleY             = 0.001f;
float scaleZ             = 0.001f;
auto cloudType           = 0;
std::string plyFile      = "heidentor_color_raw.ply";


/*uint32_t pointAmount     = 119701547;
uint32_t pointDataStride = 27;
float scaleX             = 0.01f;
float scaleY             = 0.01f;
float scaleZ             = 0.01f;
auto cloudType           = 1;
std::string plyFile      = "morrobay_fused_headerless.ply";
*/

/*
uint32_t pointAmount     = 47111095;
uint32_t pointDataStride = 27;
float scaleX             = 0.001f;
float scaleY             = 0.001f;
float scaleZ             = 0.001f;
auto cloudType           = 1;
std::string plyFile      = "lifeboat_headerless.ply";
*/
// Read in ply
auto ply = readPly (plyFile);

// Calculate BB
auto realBB  = calculateRealBB<float> (ply, pointAmount, pointDataStride);
auto cubicBB = calculateCubicBB (realBB);

auto start = std::chrono::high_resolution_clock::now ();

// Configurate and create octree
ocpi_set_cloud_type (session, cloudType);
ocpi_set_cloud_point_amount (session, pointAmount);
ocpi_set_cloud_data_stride (session, pointDataStride);
ocpi_set_cloud_scale (session, scaleX, scaleY, scaleZ);
ocpi_set_cloud_offset (session, cubicBB[0], cubicBB[1], cubicBB[2]);
ocpi_set_cloud_bb (session, cubicBB[0], cubicBB[1], cubicBB[2], cubicBB[3], cubicBB[4], cubicBB[5]);

ocpi_set_point_cloud_host (session, ply.get ());
ocpi_configure_chunking (session, chunkingGrid, mergingThreshold);
ocpi_configure_subsampling (session, subsamplingGrid, isAveraging, useReplacementScheme);

ocpi_generate_octree (session);
ocpi_export_potree (session, R"(./export)");
//ocpi_export_distribution_histogram (session, R"(./export/histogram.html)", 0);
ocpi_export_json_report (session, R"(./export/statistics.json)");
//ocpi_export_memory_report (session, R"(./export/memory_report.html)");

ocpi_destroy_session (session);
auto finish                           = std::chrono::high_resolution_clock::now ();
std::chrono::duration<double> elapsed = finish - start;
spdlog::info("Generating the octree took: {} s", elapsed.count());
}