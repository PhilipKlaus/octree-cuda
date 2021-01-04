//
// Created by KlausP on 01.11.2020.
//

#ifndef OCTREE_API_H
#define OCTREE_API_H


#include <cstdint>
#include "../OctreeLibrary/include/types.h"

#ifdef __cplusplus
#define EXPORTED_PRE extern "C"
#else
#define EXPORTED_PRE
#endif

// Define EXPORTED for any platform
#if defined _WIN32 || defined __CYGWIN__
#ifdef _DLL
// Exporting...
#ifdef __GNUC__
#define EXPORTED EXPORTED_PRE __attribute__ ((dllexport))
#else
#define EXPORTED EXPORTED_PRE __declspec(dllexport)
#endif
#else
#ifdef __GNUC__
            #define EXPORTED EXPORTED_PRE __attribute__ ((dllimport))
        #else
            #define EXPORTED                                                                                           \
                EXPORTED_PRE __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
        #endif
#endif
#else
#define EXPORTED EXPORTED_PRE
#endif


// ToDo: Add integer return value
// ToDo: Add exception handling to all function implementations
EXPORTED void ocpi_create_session (void** session, int device);
EXPORTED void ocpi_destroy_session (void* session);
EXPORTED void ocpi_set_logging_level (int level);
EXPORTED void ocpi_set_point_cloud_metadata (void* session, const PointCloudMetadata &metadata);
EXPORTED void ocpi_load_point_cloud_from_host (void* session, uint8_t *pointCloud);

EXPORTED void ocpi_configure_octree(void* session, GridSize chunkingGrid, GridSize subsamplingGrid, uint32_t mergingThreshold);
EXPORTED void ocpi_configure_memory_report(void *session, const char *filename);
EXPORTED void ocpi_configure_point_distribution_report(void *session, const char *filename, uint32_t binWidth);

EXPORTED void ocpi_generate_octree(void *session);
EXPORTED void ocpi_export_ply_nodes(void *session, const char *filename);
EXPORTED void ocpi_export_time_measurements(void *session, const char *filename);
EXPORTED void ocpi_export_octree_statistics(void *session, const char *filename);
#endif