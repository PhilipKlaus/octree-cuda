//
// Created by KlausP on 01.11.2020.
//

#include "octreeApi.h"
#include "spdlog/spdlog.h"
#include "session.h"


void ocpi_set_logging_level(int level) {
    switch(level) {
        case 1:
            spdlog::set_level(spdlog::level::info);
            break;
        case 2:
            spdlog::set_level(spdlog::level::warn);
            break;
        case 3:
            spdlog::set_level(spdlog::level::err);
            break;
        default:
            spdlog::set_level(spdlog::level::debug);
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

void ocpi_set_point_cloud_metadata (void* session, const PointCloudMetadata &metadata) {
    auto s = Session::ToSession (session);
    s->setMetadata(metadata);
}

void ocpi_load_point_cloud_from_host(void* session, uint8_t *pointCloud) {
    auto s = Session::ToSession (session);
    s->setPointCloudHost(pointCloud);
}

void ocpi_configure_octree(void* session, uint16_t globalOctreeLevel, uint32_t mergingThreshold) {
    auto s = Session::ToSession (session);
    s->setOctreeProperties(globalOctreeLevel, mergingThreshold);
}

void ocpi_generate_octree(void *session) {
    auto s = Session::ToSession (session);
    s->generateOctree();
}

void ocpi_export_octree(void *session, const char *filename) {
    auto s = Session::ToSession (session);
    s->exportOctree(filename);
}

void ocpi_configure_memory_report(void *session, const char *filename) {
    auto s = Session::ToSession (session);
    s->configureMemoryReport(filename);
}
