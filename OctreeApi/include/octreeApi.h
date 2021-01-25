//
// Created by KlausP on 01.11.2020.
//

#ifndef OCTREE_API_H
#define OCTREE_API_H

#include <api_types.h>


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

EXPORTED void ocpi_set_cloud_type (void* session, CloudType cloudType);
EXPORTED void ocpi_set_cloud_point_amount (void* session, uint32_t pointAmount);
EXPORTED void ocpi_set_cloud_data_stride (void* session, uint32_t dataStride);
EXPORTED void ocpi_set_cloud_scale_f (void* session, float x, float y, float z);
EXPORTED void ocpi_set_cloud_offset_f (void* session, float x, float y, float z);
EXPORTED void ocpi_set_cloud_bb_f(void* session, float minX, float minY, float minZ, float maxX, float maxY, float maxZ);
EXPORTED void ocpi_set_cloud_scale_d (void* session, double x, double y, double z);
EXPORTED void ocpi_set_cloud_offset_d (void* session, double x, double y, double z);
EXPORTED void ocpi_set_cloud_bb_d(void* session, double minX, double minY, double minZ, double maxX, double maxY, double maxZ);

EXPORTED void ocpi_set_point_cloud_host (void* session, uint8_t *pointCloud);

EXPORTED void ocpi_configure_chunking(void* session, GridSize chunkingGrid, uint32_t mergingThreshold);
EXPORTED void ocpi_configure_subsampling(void* session, GridSize subsamplingGrid, SubsamplingStrategy strategy);

EXPORTED void ocpi_configure_memory_report(void *session, const char *filename);
EXPORTED void ocpi_configure_point_distribution_report(void *session, const char *filename, uint32_t binWidth);
EXPORTED void ocpi_configure_json_report(void *session, const char *filename);
EXPORTED void ocpi_configure_octree_export(void *session, const char *directory);

EXPORTED void ocpi_generate_octree(void *session);

#endif