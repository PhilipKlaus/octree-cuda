#define CATCH_CONFIG_RUNNER // This tells Catch to provide a main() - only do this in one cpp file
#include "catch2/catch.hpp"
#include "spdlog/spdlog.h"

int main (int argc, char* argv[])
{
    // global setup...
    spdlog::set_level (spdlog::level::err);

    int result = Catch::Session ().run (argc, argv);

    // global clean-up...

    return result;
}