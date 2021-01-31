//
// Created by KlausP on 01.11.2020.
//

#include "octreeApi.h"
#include "session.h"
#include "spdlog/spdlog.h"


void ocpi_set_logging_level (int level)
{
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
}

void ocpi_create_session (void** session, int device)
{
    *session = new Session (device);
}

void ocpi_destroy_session (void* session)
{
    auto s = Session::ToSession (session);
    delete s;
}

void ocpi_set_point_cloud_host (void* session, uint8_t* pointCloud)
{
    auto s = Session::ToSession (session);
    s->setPointCloudHost (pointCloud);
}

void ocpi_generate_octree (void* session)
{
    auto s = Session::ToSession (session);
    s->generateOctree ();
}

void ocpi_configure_octree_export (void* session, const char* filename)
{
    auto s = Session::ToSession (session);
    s->configureOctreeExport (filename);
}

void ocpi_configure_chunking (void* session, uint32_t chunkingGrid, uint32_t mergingThreshold)
{
    auto s = Session::ToSession (session);
    s->configureChunking (chunkingGrid, mergingThreshold);
}

void ocpi_configure_subsampling (void* session, uint32_t subsamplingGrid, uint8_t strategy)
{
    auto s = Session::ToSession (session);
    s->configureSubsampling (subsamplingGrid, strategy);
}

void ocpi_configure_memory_report (void* session, const char* filename)
{
    auto s = Session::ToSession (session);
    s->configureMemoryReport (filename);
}

void ocpi_configure_json_report (void* session, const char* filename)
{
    auto s = Session::ToSession (session);
    s->configureJsonReport (filename);
}

void ocpi_configure_point_distribution_report (void* session, const char* filename, uint32_t binWidth)
{
    auto s = Session::ToSession (session);
    s->configurePointDistributionReport (filename, binWidth);
}

void ocpi_set_cloud_type (void* session, uint8_t cloudType)
{
    auto s = Session::ToSession (session);
    s->setCloudType (cloudType);
}

void ocpi_set_cloud_point_amount (void* session, uint32_t pointAmount)
{
    auto s = Session::ToSession (session);
    s->setCloudPointAmount (pointAmount);
}

void ocpi_set_cloud_data_stride (void* session, uint32_t dataStride)
{
    auto s = Session::ToSession (session);
    s->setCloudDataStride (dataStride);
}

void ocpi_set_cloud_scale_f (void* session, float x, float y, float z)
{
    auto s = Session::ToSession (session);
    s->setCloudScaleF (x, y, z);
}

void ocpi_set_cloud_offset_f (void* session, float x, float y, float z)
{
    auto s = Session::ToSession (session);
    s->setCloudOffsetF (x, y, z);
}

void ocpi_set_cloud_bb_f (void* session, float minX, float minY, float minZ, float maxX, float maxY, float maxZ)
{
    auto s = Session::ToSession (session);
    s->setCloudBoundingBoxF (minX, minY, minZ, maxX, maxY, maxZ);
}

void ocpi_set_cloud_offset_d (void* session, double x, double y, double z)
{
    auto s = Session::ToSession (session);
    s->setCloudOffsetD (x, y, z);
}

void ocpi_set_cloud_bb_d (void* session, double minX, double minY, double minZ, double maxX, double maxY, double maxZ)
{
    auto s = Session::ToSession (session);
    s->setCloudBoundingBoxD (minX, minY, minZ, maxX, maxY, maxZ);
}
