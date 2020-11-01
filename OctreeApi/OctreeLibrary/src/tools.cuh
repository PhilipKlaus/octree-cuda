#ifndef OCTREE_TOOLS
#define OCTREE_TOOLS

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <memory>
#include "cudaArray.h"
#include "types.h"

using namespace std;


namespace tools {

    unique_ptr<CudaArray<Vector3>> generate_point_cloud_cuboid(uint32_t sideLength);
    void printKernelDimensions(dim3 block, dim3 grid);
    void create1DKernel(dim3 &block, dim3 &grid, uint32_t pointCount);

    __device__ Vector3 subtract(const Vector3 &a,const Vector3 &b);
    __device__ uint32_t calculateGridIndex(const Vector3 &point, PointCloudMetadata const &metadata, uint16_t gridSize);

};

#endif