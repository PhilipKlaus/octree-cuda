#include "tools.cuh"
#include "types.h"
#include "defines.cuh"


__global__ void kernel_point_cloud_cuboid(Vector3 *out, unsigned int n, unsigned int side) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= n) {
        return;
    }

    auto xy = side * side;
    auto z = index / xy;
    auto y = (index - (z * xy)) / side;
    auto x = (index - (z * xy)) % side;

    out[index].x = x + 0.5;
    out[index].y = y + 0.5;
    out[index].z = z + 0.5;
}


namespace tools {

    __device__ Vector3 subtract(const Vector3 &a,const Vector3 &b) {
        return {
                a.x - b.x,
                a.y - b.y,
                a.z - b.z
        };
    }

    __device__ uint64_t calculateGridIndex(const Vector3 &point, PointCloudMetadata const &metadata, uint16_t gridSize) {

        // See OctreeConverter : chunker_countsort_laszip.cpp :131

        float dGridSize = gridSize;
        auto X = static_cast<int32_t>((point.x - metadata.cloudOffset.x) / metadata.scale.x);
        auto Y = static_cast<int32_t>((point.y - metadata.cloudOffset.y) / metadata.scale.y);
        auto Z = static_cast<int32_t>((point.z - metadata.cloudOffset.z) / metadata.scale.z);
        auto size = tools::subtract(metadata.boundingBox.maximum, metadata.boundingBox.minimum);

        float ux =
                (static_cast<float>(X) * metadata.scale.x + metadata.cloudOffset.x - metadata.boundingBox.minimum.x)
                / size.x;
        float uy =
                (static_cast<float>(Y) * metadata.scale.y + metadata.cloudOffset.y - metadata.boundingBox.minimum.y)
                / size.y;
        float uz =
                (static_cast<float>(Z) * metadata.scale.z + metadata.cloudOffset.z - metadata.boundingBox.minimum.z)
                / size.z;

        uint64_t ix = static_cast<int64_t>( fmin (dGridSize * ux, dGridSize - 1.0f));
        uint64_t iy = static_cast<int64_t>( fmin (dGridSize * uy, dGridSize - 1.0f));
        uint64_t iz = static_cast<int64_t>( fmin (dGridSize * uz, dGridSize - 1.0f));

        return ix + iy * gridSize + iz * gridSize * gridSize;
    }

    void create1DKernel(dim3 &block, dim3 &grid, uint32_t pointCount) {

        auto blocks = ceil(static_cast<float>(pointCount) / BLOCK_SIZE_MAX);
        auto gridX = blocks < GRID_SIZE_MAX ? blocks : ceil(blocks / GRID_SIZE_MAX);
        auto gridY = ceil(blocks / gridX);


        block = dim3(BLOCK_SIZE_MAX, 1, 1);
        grid = dim3 (static_cast<unsigned int>(gridX), static_cast<unsigned int>(gridY), 1);
        printKernelDimensions(block, grid);
    }

    unique_ptr<CudaArray<Vector3>> generate_point_cloud_cuboid(unsigned int sideLength) {

        auto pointAmount = sideLength * sideLength * sideLength;
        auto data = std::make_unique<CudaArray<Vector3>>(pointAmount);

        auto blocks = ceil(pointAmount / 1024.f);
        auto gridX = blocks < GRID_SIZE_MAX ? blocks : ceil(blocks / GRID_SIZE_MAX);
        auto gridY = ceil(blocks / gridX);

        dim3 block(BLOCK_SIZE_MAX, 1, 1);
        dim3 grid(static_cast<unsigned int>(gridX), static_cast<unsigned int>(gridY), 1);
        printKernelDimensions(block, grid);

        kernel_point_cloud_cuboid <<<  grid, block >>> (data->devicePointer(), pointAmount, sideLength);
        return data;
    }

    void printKernelDimensions(dim3 block, dim3 grid) {
        spdlog::debug(
                "Launching kernel with dimensions: "
                "block [{}, {}, {}] | grid[{}, {}, {}]",
                block.x, block.y, block.z, grid.x, grid.y, grid.z
        );
    }

}



