#include "argparser.h"
#include "boundingbox.h"
#include "octreeApi.h"
#include "spdlog/spdlog.h"

#include <filesystem>
#include <fstream>

using namespace std;
namespace fs = std::filesystem;

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

int main (int argc, char** argv)
{
#ifndef NDEBUG
    spdlog::set_level (spdlog::level::debug);
    ocpi_set_logging_level (0);
#else
    spdlog::set_level (spdlog::level::info);
    ocpi_set_logging_level (1);
#endif

    Input input = {};

    try
    {
        parseArguments (argc, argv, input);
    }
    catch (cxxopts::OptionParseException& exc)
    {
        spdlog::error ("{}", exc.what ());
        exit (1);
    }

    printInputConfig (input);

    // Create output dir if not existing
    if (!fs::exists (input.outputPath))
    {
        fs::create_directories (input.outputPath);
    }

    void* session;
    ocpi_create_session (&session, 0);

    spdlog::info ("--------------------------------");
    auto start = std::chrono::high_resolution_clock::now ();

    // Read in ply
    auto ply = readPly (input.inputFile);

    // Calculate BB
    std::vector<double> realBB;
    std::vector<double> cubicBB;

    if (input.cloudType == 0)
    {
        realBB = calculateRealBB<float> (ply, input.pointAmount, input.pointDataStride);
    }
    else
    {
        realBB = calculateRealBB<double> (ply, input.pointAmount, input.pointDataStride);
    }
    cubicBB = calculateCubicBB (realBB);

    auto finish                           = std::chrono::high_resolution_clock::now ();
    std::chrono::duration<double> elapsed = finish - start;
    spdlog::info ("Reading cloud and calc bounding box took: {} [s]", elapsed.count ());
    spdlog::info ("--------------------------------");

    // Configurate and create octree
    ocpi_set_cloud_type (session, input.cloudType);
    ocpi_set_cloud_point_amount (session, input.pointAmount);
    ocpi_set_cloud_data_stride (session, input.pointDataStride);
    ocpi_set_cloud_scale (session, input.scale, input.scale, input.scale);
    ocpi_set_cloud_offset (session, cubicBB[0], cubicBB[1], cubicBB[2]);
    ocpi_set_cloud_bb (session, cubicBB[0], cubicBB[1], cubicBB[2], cubicBB[3], cubicBB[4], cubicBB[5]);

    ocpi_set_point_cloud_host (session, ply.get ());
    ocpi_configure_chunking (session, input.chunkingGrid, input.mergingThreshold, input.outputFactor);
    ocpi_configure_subsampling (
            session,
            input.subsamplingGrid,
            input.isIntraCellAveraging,
            input.isInterCellAveraging,
            input.useReplacementScheme,
            input.performRandomSubsampling);

    ocpi_init_octree (session);
    ocpi_generate_octree (session);
    ocpi_export_potree (session, input.outputPath.c_str ());

    ocpi_export_distribution_histogram (session, (input.outputPath + "/point_distribution.html").c_str (), 0);
    ocpi_export_json_report (session, (input.outputPath + "/statistics.json").c_str ());
    ocpi_export_memory_report (session, (input.outputPath + "/memory_report.html").c_str ());

    ocpi_destroy_session (session);
}