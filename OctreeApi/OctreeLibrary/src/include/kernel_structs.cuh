/**
 * @file kernel_structs.cuh
 * @author Philip Klaus
 * @brief Contains structs which are passed to CUDA kernels
 */

#pragma once

#include <cstdint>
#include "metadata.cuh"

namespace KernelStructs {


/**
 * A struct which stores data necessary for gridding (mapping 3D points to a 3D grid)
 */
struct Gridding
{
    uint32_t gridSize;     ///< The size of one grid size (e.g. 128 -> 128 x 128 128)
    double bbSize;         ///< The length of one side of the cubic cloud bounding box
    Vector3<double> bbMin; ///< The bounding box minimum
};

struct Cloud
{
    uint8_t* raw;        ///< The raw point cloud data
    uint32_t points;     ///< The amount of points wihtin the point cloud
    uint32_t dataStride; ///< The point cloud data stride
};

struct NodeOutput {
    uint32_t* pointCount;
    uint32_t* pointOffset;
};
} // namespace KernelStructs