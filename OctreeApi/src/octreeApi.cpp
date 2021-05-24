//
// Created by KlausP on 01.11.2020.
//

#include "octreeApi.h"
#include "session.h"
#include "spdlog/spdlog.h"

template <typename Function, typename... Args>
int Execute (Function function, Args... args)
{
    try
    {
        function (args...);
        return SUCCESS;
    }
    catch (...)
    {
        return UNEXPECTED_ERROR;
    }
}

int ocpi_set_logging_level (int level)
{
    return Execute ([&] () {
        switch (level)
        {
        case 1:
            spdlog::set_level (spdlog::level::info);
            break;
        case 2:
            spdlog::set_level (spdlog::level::warn);
            break;
        case 3:
            spdlog::set_level (spdlog::level::err);
            break;
        default:
            spdlog::set_level (spdlog::level::debug);
            break;
        }
    });
}

int ocpi_create_session (void** session, int device)
{
    return Execute ([&] () { *session = new Session (device); });
}

int ocpi_destroy_session (void* session)
{
    return Execute ([&] () {
        auto s = Session::ToSession (session);
        delete s;
    });
}

int ocpi_set_point_cloud_host (void* session, uint8_t* pointCloud)
{
    return Execute ([&] () {
        auto s = Session::ToSession (session);
        s->setPointCloudHost (pointCloud);
    });
}

int ocpi_init_octree (void* session)
{
    return Execute ([&] () {
        auto s = Session::ToSession (session);
        s->initOctree ();
    });
}

int ocpi_generate_octree (void* session)
{
    return Execute ([&] () {
        auto s = Session::ToSession (session);
        s->generateOctree ();
    });
}

int ocpi_export_potree (void* session, const char* filename)
{
    return Execute ([&] () {
        auto s = Session::ToSession (session);
        s->exportPotree (filename);
    });
}

int ocpi_configure_chunking (void* session, uint32_t chunkingGrid, uint32_t mergingThreshold, float outputFactor)
{
    return Execute ([&] () {
        auto s = Session::ToSession (session);
        s->configureChunking (chunkingGrid, mergingThreshold, outputFactor);
    });
}

int ocpi_configure_subsampling (
        void* session, uint32_t subsamplingGrid, bool averaging, bool replacementScheme, bool useRandomSubsampling)
{
    return Execute ([&] () {
        auto s = Session::ToSession (session);
        s->configureSubsampling (subsamplingGrid, averaging, replacementScheme, useRandomSubsampling);
    });
}

int ocpi_export_memory_report (void* session, const char* filename)
{
    return Execute ([&] () {
        auto s = Session::ToSession (session);
        s->exportMemoryReport (filename);
    });
}

int ocpi_export_json_report (void* session, const char* filename)
{
    return Execute ([&] () {
        auto s = Session::ToSession (session);
        s->exportJsonReport (filename);
    });
}

int ocpi_export_distribution_histogram (void* session, const char* filename, uint32_t binWidth)
{
    return Execute ([&] () {
        auto s = Session::ToSession (session);
        s->exportDistributionHistogram (filename, binWidth);
    });
}

int ocpi_set_cloud_type (void* session, uint8_t cloudType)
{
    return Execute ([&] () {
        auto s = Session::ToSession (session);
        s->setCloudType (cloudType);
    });
}

int ocpi_set_cloud_point_amount (void* session, uint32_t pointAmount)
{
    return Execute ([&] () {
        auto s = Session::ToSession (session);
        s->setCloudPointAmount (pointAmount);
    });
}

int ocpi_set_cloud_data_stride (void* session, uint32_t dataStride)
{
    return Execute ([&] () {
        auto s = Session::ToSession (session);
        s->setCloudDataStride (dataStride);
    });
}

int ocpi_set_cloud_scale (void* session, double x, double y, double z)
{
    return Execute ([&] () {
        auto s = Session::ToSession (session);
        s->setCloudScale (x, y, z);
    });
}

int ocpi_set_cloud_offset (void* session, double x, double y, double z)
{
    return Execute ([&] () {
        auto s = Session::ToSession (session);
        s->setCloudOffset (x, y, z);
    });
}

int ocpi_set_cloud_bb (void* session, double minX, double minY, double minZ, double maxX, double maxY, double maxZ)
{
    return Execute ([&] () {
        auto s = Session::ToSession (session);
        s->setCloudBoundingBox (minX, minY, minZ, maxX, maxY, maxZ);
    });
}