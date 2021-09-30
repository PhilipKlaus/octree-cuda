#pragma once

#include "cxxopts/cxxopts.hpp"
#include "spdlog/spdlog.h"

struct Input
{
    std::string inputFile;
    std::string outputPath;
    uint32_t pointAmount;
    uint32_t pointDataStride;
    float scale;
    bool isIntraCellAveraging;
    bool isInterCellAveraging;
    bool useReplacementScheme;
    bool performRandomSubsampling;
    uint32_t chunkingGrid;
    uint32_t subsamplingGrid;
    uint32_t mergingThreshold;
    uint8_t cloudType;
    float outputFactor;
};

cxxopts::Options createOptions ()
{
    cxxopts::Options options ("PotreeConverterGPU", "Generates Potree compatible LOD structures on the GPU");
    options.add_options () ("f,file", "File name point cloud", cxxopts::value<std::string> ()) (
            "a,averaging-intra", "Apply intra-cell color averaging", cxxopts::value<bool> ()->default_value ("false")) (
            "i,averaging-inter", "Apply inter-cell color averaging", cxxopts::value<bool> ()->default_value ("false")) (
            "r,random",
            "Perform Random-Subsampling, otherwise First-Point-Subsampling is applied",
            cxxopts::value<bool> ()->default_value ("false")) (
            "o,output", "Output path for the Potree data", cxxopts::value<std::string> ()) (
            "p,points", "Point amount of the cloud", cxxopts::value<uint32_t> ()) (
            "t,type", R"(The datatype of the cloud coordinates: "float" / "double")", cxxopts::value<std::string> ()) (
            "d,data", "Data infos for stride and scale: [float, float]", cxxopts::value<std::vector<float>> ()) (
            "g,grids",
            "Grid sizes for chunking and subsampling: [int, int]",
            cxxopts::value<std::vector<uint32_t>> ()) (
            "m,merge_threshold", "The merging threshold", cxxopts::value<uint32_t> ()->default_value ("10000")) (
            "e,estimated_output",
            "The estimated output point amount factor",
            cxxopts::value<float> ()->default_value ("2.2")) ("h,help", "Print usage");
    return options;
}

void checkForMissingParameters (const cxxopts::ParseResult& result, cxxopts::Options& options)
{
    if (result.count ("help"))
    {
        std::cout << options.help () << std::endl;
        exit (0);
    }
    if (!result.count ("file"))
    {
        spdlog::error ("No input file (-f) specified");
        exit (-1);
    }

    if (!result.count ("points"))
    {
        spdlog::error ("No point amount (-p) specified");
        exit (-1);
    }

    if (!result.count ("type"))
    {
        spdlog::error ("No coordinate datatype (-t) specified");
        exit (-1);
    }

    if (!result.count ("data"))
    {
        spdlog::error ("No data info (-d) specified");
        exit (-1);
    }
}

void checkForValidParameters (const cxxopts::ParseResult& result)
{
    if (result["data"].as<std::vector<float>> ().size () < 2)
    {
        spdlog::error ("Data info has to contain stride and scale");
        exit (-1);
    }

    if (result["type"].as<std::string> () != "float" && result["type"].as<std::string> () != "double")
    {
        spdlog::error (R"(Data type must be "float" or "double")");
        exit (-1);
    }

    if (result["averaging-intra"].as<bool> () && result["averaging-inter"].as<bool> ())
    {
        spdlog::error (R"(Either intra-cell (-a) or inter-cell (-i) can be applied")");
        exit (-1);
    }

    std::vector<uint32_t> grids;
    if (result.count ("grids"))
    {
        if (result["grids"].as<std::vector<uint32_t>> ().size () < 2)
        {
            spdlog::error ("Grid sizes have to contain [chunking_grid, subsampling_grid]");
            exit (-1);
        }
    }
}

void printInputConfig (const Input& input)
{
    spdlog::info ("--------------------------------");
    spdlog::info ("Input file: {}", input.inputFile);
    spdlog::info ("Output path: {}", input.outputPath);
    spdlog::info ("pointAmount: {}", input.pointAmount);
    spdlog::info ("cloudType: {}", input.cloudType);
    spdlog::info ("dataStride: {}", input.pointDataStride);
    spdlog::info ("scale: {}", input.scale);
    spdlog::info ("chunkingGrid: {}", input.chunkingGrid);
    spdlog::info ("subsamplingGrid: {}", input.subsamplingGrid);
    spdlog::info ("mergingThreshold: {}", input.mergingThreshold);
    spdlog::info ("outputFactor: {}", input.outputFactor);
    spdlog::info ("perform intra-cell averaging: {}", input.isIntraCellAveraging);
    spdlog::info ("perform inter-cell averaging: {}", input.isInterCellAveraging);
    spdlog::info ("Subsampling method: {}", input.performRandomSubsampling ? "random" : "first-point");
}

void parseInput (Input& input, const cxxopts::ParseResult& result)
{
    auto dataInfo = result["data"].as<std::vector<float>> ();
    auto grids    = result["grids"].as<std::vector<uint32_t>> ();

    input.inputFile                = result["file"].as<std::string> ();
    input.outputPath               = result["output"].as<std::string> ();
    input.pointAmount              = result["points"].as<uint32_t> ();
    input.pointDataStride          = static_cast<uint32_t> (dataInfo[0]);
    input.scale                    = dataInfo[1];
    input.isIntraCellAveraging     = result["averaging-intra"].as<bool> ();
    input.isInterCellAveraging     = result["averaging-inter"].as<bool> ();
    input.performRandomSubsampling = result["random"].as<bool> ();
    input.chunkingGrid             = grids[0];
    input.subsamplingGrid          = grids[1];
    input.mergingThreshold         = result["merge_threshold"].as<uint32_t> ();
    input.outputFactor             = result["estimated_output"].as<float> ();
    input.cloudType                = result["type"].as<std::string> () == "float" ? 0 : 1;
}

void parseArguments (int argc, char** argv, Input& input)
{
    auto options = createOptions ();
    auto result  = options.parse (argc, argv);

    checkForMissingParameters (result, options);
    checkForValidParameters (result);
    parseInput (input, result);
}