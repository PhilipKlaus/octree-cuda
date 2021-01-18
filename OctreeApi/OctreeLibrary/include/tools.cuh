#pragma once


#include <cuda_runtime_api.h>
#include <cuda.h>
#include <memory>
#include "cudaArray.h"
#include "global_types.h"

using namespace std;


namespace tools {

uint32_t getOctreeLevel(GridSize gridSize);
    uint32_t getOctreeGrid(uint32_t octreeLevel);
    uint32_t getNodeAmount(uint32_t octreeLevel);
    uint32_t getNodeOffset(uint32_t octreeLevel, uint32_t octreeDepth);

    unique_ptr<CudaArray<uint8_t>> generate_point_cloud_cuboid(uint32_t sideLength, PointCloudMetadata &metadata);
    void printKernelDimensions(dim3 block, dim3 grid);
    void create1DKernel(dim3 &block, dim3 &grid, uint32_t pointCount);

    __host__ __device__ void mapFromDenseIdxToDenseCoordinates(Vector3<uint32_t> &coordinates, uint32_t denseVoxelIdx, uint32_t level);


    // See OctreeConverter : chunker_countsort_laszip.cpp :131
    template <typename coordinateType>
    __device__ uint32_t calculateGridIndex(const Vector3<coordinateType> *point, PointCloudMetadata const &metadata, uint32_t gridSize) {

        double sizeX = metadata.boundingBox.maximum.x - metadata.boundingBox.minimum.x;
        double sizeY = metadata.boundingBox.maximum.y - metadata.boundingBox.minimum.y;
        double sizeZ = metadata.boundingBox.maximum.z - metadata.boundingBox.minimum.z;

        double uX = (point->x - metadata.boundingBox.minimum.x) / (sizeX / gridSize);
        double uY = (point->y - metadata.boundingBox.minimum.y) / (sizeY / gridSize);
        double uZ = (point->z - metadata.boundingBox.minimum.z) / (sizeZ / gridSize);

        uint64_t ix = static_cast<int64_t >( fmin (uX, gridSize - 1.0));
        uint64_t iy = static_cast<int64_t>( fmin (uY, gridSize - 1.0));
        uint64_t iz = static_cast<int64_t>( fmin (uZ, gridSize - 1.0));

        return static_cast<uint32_t >(ix + iy * gridSize + iz * gridSize * gridSize);
    }
};
