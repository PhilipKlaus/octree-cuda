/**
 * @file metadata.cuh
 * @author Philip Klaus
 * @brief Contains metadatd definitions
 */

#pragma once

#include <cstdint>


// ToDo: Check if pragma is necessary
#pragma pack(push, 1)
template <typename T>
struct Vector3
{
    T x, y, z;
};
#pragma pack(pop)


/**
 * @brief The data type of the point cloud
 */
enum CloudType
{
    CLOUD_FLOAT_UINT8_T,
    CLOUD_DOUBLE_UINT8_T
};

/**
 * @brief The memory type of the point cloud
 */
enum CloudMemory
{
    CLOUD_HOST,
    ClOUD_DEVICE
};

/**
 * @brief Represents a cloud bounding box
 */
struct BoundingBox
{
    Vector3<double> min; ///< The minimum position of the bounding box e.g. -1, -1, -1
    Vector3<double> max; ///< The maximum position of the bounding box e.g. 1, 1, 1
};

/**
 * @brief Information about the point cloud
 */
struct PointCloudInfo
{
    uint32_t pointAmount;        ///< The amount of points in the cloud
    uint32_t pointDataStride;    ///< The datastride e.g.: x(float), y(float), z(float) = 12 bytes
    BoundingBox bbCubic;         ///< The cubic bounding box (equal side length)
    Vector3<double> cloudOffset; ///< The offset of the point cloud
    Vector3<double> scale;       ///< The scale of the cloud
    CloudType cloudType;         ///< The cloud data type
    CloudMemory memoryType;      ///< The cloud memory type

    /**
     * Calculate the side length of the cubic bounding box
     * @return side length of the cubic bounding box
     */
    double cubicSize () const
    {
        return bbCubic.max.x - bbCubic.min.x;
    }
};

/**
 * @brief Information about octree processing
 */
struct ProcessingInfo
{
    uint32_t subsamplingGrid;  ///< The size of the subsampling grid (e.g. 128 -> 128 x 128 128)
    uint32_t chunkingGrid;     ///< The size of the chunking grid (e.g. 128 -> 128 x 128 128)
    uint32_t mergingThreshold; ///< The threshold for the cell merging during the chunking phase
    bool performAveraging;     ///< Determines if averaging should be performed
    bool useReplacementScheme; ///< Determines if the replacement scheme should be applied
};

/**
 * @brief Information about the generated octree
 */
struct OctreeInfo
{
    uint32_t depth;               ///< The depth of the octree (amount of levels)
    uint8_t maxLeafDepth;         ///< Maximum depth of a leaf node
    uint32_t leafNodeAmount;      ///< The amount of leaf nodes
    uint32_t parentNodeAmount;    ///< The amount of parent nodes
    float meanPointsPerLeafNode;  ///< Mean point amount per leaf node
    float stdevPointsPerLeafNode; ///< Standard deviation of the point amount per leaf node
    uint32_t minPointsPerNode;    ///< Minimum number of points per leaf node
    uint32_t maxPointsPerNode;    ///< Maximum number of points per leaf node
    uint32_t nodeAmountSparse;    ///< The sparse node amount (before node merging)
    uint32_t nodeAmountDense;     ///< The theoretical dense node amount
};
