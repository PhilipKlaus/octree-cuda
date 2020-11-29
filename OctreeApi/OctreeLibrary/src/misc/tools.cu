#include "tools.cuh"
#include "defines.cuh"


__global__ void kernel_point_cloud_cuboid(uint8_t *out, uint32_t n, uint32_t side) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= n) {
        return;
    }

    auto xy = side * side;
    auto z = index / xy;
    auto y = (index - (z * xy)) / side;
    auto x = (index - (z * xy)) % side;

    reinterpret_cast<Vector3*>(out + index * 12)->x = x + 0.5;
    reinterpret_cast<Vector3*>(out + index * 12)->y = y + 0.5;
    reinterpret_cast<Vector3*>(out + index * 12)->z = z + 0.5;
}


namespace tools {

    uint32_t getOctreeLevel(OctreeTypes::GridSize gridSize) {
        switch (gridSize) {
            case OctreeTypes::GRID_512:
                return 9;
            case OctreeTypes::GRID_256:
                return 8;
            case OctreeTypes::GRID_128:
                return 7;
            case OctreeTypes::GRID_64:
                return 6;
            case OctreeTypes::GRID_32:
                return 5;
            case OctreeTypes::GRID_16:
                return 4;
            case OctreeTypes::GRID_8:
                return 3;
            case OctreeTypes::GRID_4:
                return 2;
            case OctreeTypes::GRID_2:
                return 1;
            default:
                return 0;
        }
    }

    uint32_t getOctreeGrid(uint32_t octreeLevel) {
        switch (octreeLevel) {
            case 9:
                return 512;
            case 8:
                return 256;
            case 7:
                return 128;
            case 6:
                return 64;
            case 5:
                return 32;
            case 4:
                return 16;
            case 3:
                return 8;
            case 2:
                return 4;
            case 1:
                return 2;
            default:
                return 1;
        }
    }

    uint32_t getNodeAmount(uint32_t octreeLevel) {
        switch (octreeLevel) {
            case 9: // 512
                return 134217728;
            case 8: // 256
                return 16777216;
            case 7: // 128
                return 2097152;
            case 6: // 64
                return 262144;
            case 5: // 32
                return 32768;
            case 4: // 16
                return 4096;
            case 3: // 8
                return 512;
            case 2: // 4
                return 64;
            case 1: // 2
                return 8;
            default: // 1
                return 1;
        }
    }

    uint32_t getNodeOffset(uint32_t octreeLevel, uint32_t octreeDepth) {
        uint32_t offset = 0;
        for(uint32_t i = octreeDepth; i < octreeLevel; --i) {
            offset += getNodeAmount(i);
        }
        return offset;
    }

    __host__ __device__ Vector3 subtract(const Vector3 &a,const Vector3 &b) {
        return {
                a.x - b.x,
                a.y - b.y,
                a.z - b.z
        };
    }

    __device__ uint32_t calculateGridIndex(const Vector3 *point, PointCloudMetadata const &metadata, uint16_t gridSize) {

        // See OctreeConverter : chunker_countsort_laszip.cpp :131

        float dGridSize = gridSize;
        auto X = static_cast<int32_t>((point->x - metadata.cloudOffset.x) / metadata.scale.x);
        auto Y = static_cast<int32_t>((point->y - metadata.cloudOffset.y) / metadata.scale.y);
        auto Z = static_cast<int32_t>((point->z - metadata.cloudOffset.z) / metadata.scale.z);
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

        uint32_t ix = static_cast<int64_t>( fmin (dGridSize * ux, dGridSize - 1.0f));
        uint32_t iy = static_cast<int64_t>( fmin (dGridSize * uy, dGridSize - 1.0f));
        uint32_t iz = static_cast<int64_t>( fmin (dGridSize * uz, dGridSize - 1.0f));

        return ix + iy * gridSize + iz * gridSize * gridSize;
    }

    void create1DKernel(dim3 &block, dim3 &grid, uint32_t pointCount) {

        auto blocks = ceil(static_cast<double>(pointCount) / BLOCK_SIZE_MAX);
        auto gridX = blocks < GRID_SIZE_MAX ? blocks : ceil(blocks / GRID_SIZE_MAX);
        auto gridY = ceil(blocks / gridX);


        block = dim3(BLOCK_SIZE_MAX, 1, 1);
        grid = dim3 (static_cast<unsigned int>(gridX), static_cast<unsigned int>(gridY), 1);
        printKernelDimensions(block, grid);
    }

    unique_ptr<CudaArray<uint8_t>> generate_point_cloud_cuboid(uint32_t sideLength, PointCloudMetadata &metadata) {

        float boundingBoxMax = static_cast<float>(sideLength) - 0.5f;
        metadata.pointAmount = static_cast<uint32_t>(pow(sideLength, 3.f));
        metadata.boundingBox.minimum = Vector3 {0.5, 0.5, 0.5};
        metadata.boundingBox.maximum = Vector3 {boundingBoxMax, boundingBoxMax, boundingBoxMax};
        metadata.cloudOffset = Vector3 {0.5, 0.5, 0.5};
        metadata.scale = {1.f, 1.f, 1.f};
        metadata.pointDataStride = 12;

        auto pointAmount = sideLength * sideLength * sideLength;
        auto data = std::make_unique<CudaArray<uint8_t>>(pointAmount * 12, "cuboid");

        auto blocks = ceil(pointAmount / 1024.f);
        auto gridX = blocks < GRID_SIZE_MAX ? blocks : ceil(blocks / GRID_SIZE_MAX);
        auto gridY = ceil(blocks / gridX);

        dim3 block(BLOCK_SIZE_MAX, 1, 1);
        dim3 grid(static_cast<uint32_t>(gridX), static_cast<uint32_t>(gridY), 1);
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

    __host__ __device__ void mapFromDenseIdxToDenseCoordinates(
            Vector3i &coordinates,
            uint32_t denseVoxelIdx,
            uint32_t gridSizeLength) {

        auto xy = gridSizeLength * gridSizeLength;
        coordinates.z = denseVoxelIdx / xy;
        coordinates.y = (denseVoxelIdx - (coordinates.z * xy)) / gridSizeLength;
        coordinates.x = (denseVoxelIdx - (coordinates.z * xy)) % gridSizeLength;
    }
}



