//
// Created by KlausP on 01.11.2020.
//

#ifndef OCTREE_API_H
#define OCTREE_API_H


#include <cstdint>

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

struct Vector3
{
    float x, y, z;
};

struct Chunk {
    uint32_t pointCount;        // How many points does this chunk have
    uint32_t parentChunkIndex;  // Determines the INDEX of the parent CHUNK in the GRID - Only needed during Merging
    bool isFinished;            // Is this chunk finished (= not mergeable anymore)
    uint32_t chunkDataIndex;    // Determines the INDEX in the chunk data array -> for storing point data
    int childrenChunks[8];      // The INDICES of the children chunks in the GRID
    uint32_t childrenChunksCount;
};

struct BoundingBox {
    Vector3 minimum;
    Vector3 maximum;
};

struct PointCloudMetadata {
    uint32_t pointAmount;
    BoundingBox boundingBox;
    Vector3 cloudOffset;
    Vector3 scale;
};


EXPORTED void ocpi_create_session (void** session, int device);
EXPORTED void ocpi_destroy_session (void* session);
EXPORTED void ocpi_set_logging_level (int level);
EXPORTED void ocpi_set_point_cloud_metadata (void* session, const PointCloudMetadata &metadata);
EXPORTED void ocpi_load_point_cloud_from_host (void* session, uint8_t *pointCloud);
EXPORTED void ocpi_configure_octree(void* session, uint16_t globalOctreeLevel, uint32_t mergingThreshold);
EXPORTED void ocpi_generate_octree(void *session);
#endif