#ifndef OCTREE_TOOLS
#define OCTREE_TOOLS

#include <cuda_runtime_api.h>
#include <cuda.h>
# include <iostream>
# include <memory>
#include "types.h"
#include <functional>

using namespace std;

constexpr unsigned int BLOCK_SIZE_MAX = 1024;
constexpr unsigned int GRID_SIZE_MAX = 65535;


unique_ptr<CudaArray<Vector3>> generate_point_cloud_cuboid(unsigned int sideLength);

static inline unsigned int divUp (const int64_t a, const int64_t b)
{
    return static_cast<unsigned int> ((a % b != 0) ? (a / b + 1) : (a / b));
}

void create1DKernel(dim3 &block, dim3 &grid, uint32_t pointCount);
unique_ptr<CudaArray<Vector3>> exportToPly(unique_ptr<CudaArray<Vector3>> pointCloud, std::string file_name);
#endif