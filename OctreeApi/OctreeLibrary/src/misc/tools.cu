#include "../include/tools.cuh"
#include "defines.cuh"
#include "metadata.cuh"
#include "types.cuh"


__global__ void kernel_point_cloud_cuboid (uint8_t* out, uint32_t n, uint32_t side)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n)
    {
        return;
    }

    auto xy = side * side;
    auto z  = index / xy;
    auto y  = (index - (z * xy)) / side;
    auto x  = (index - (z * xy)) % side;

    reinterpret_cast<Vector3<float>*> (out + index * 12)->x = x + 0.5;
    reinterpret_cast<Vector3<float>*> (out + index * 12)->y = y + 0.5;
    reinterpret_cast<Vector3<float>*> (out + index * 12)->z = z + 0.5;
}


namespace tools {


void calculateChildMinBB (
        Vector3<double>& childBBMin, const Vector3<double>& parentBBMin, uint8_t localIdx, double childBBSide)
{
    switch (localIdx)
    {
    case 0:
        childBBMin = {parentBBMin.x, parentBBMin.y, parentBBMin.z};
        break;
    case 1:
        childBBMin.z += childBBSide;
        break;
    case 2:
        childBBMin.y += childBBSide;
        break;
    case 3:
        childBBMin = {parentBBMin.x, parentBBMin.y + childBBSide, parentBBMin.z + childBBSide};
        break;
    case 4:
        childBBMin.x += childBBSide;
        break;
    case 5:
        childBBMin = {parentBBMin.x + childBBSide, parentBBMin.y, parentBBMin.z + childBBSide};
        break;
    case 6:
        childBBMin = {parentBBMin.x + childBBSide, parentBBMin.y + childBBSide, parentBBMin.z};
        break;
    case 7:
        childBBMin = {parentBBMin.x + childBBSide, parentBBMin.y + childBBSide, parentBBMin.z + childBBSide};
        break;
    default:
        break;
    }
}

uint8_t getOctreeLevel (uint32_t gridSize)
{
    switch (gridSize)
    {
    case 512:
        return 9;
    case 256:
        return 8;
    case 128:
        return 7;
    case 64:
        return 6;
    case 32:
        return 5;
    case 16:
        return 4;
    case 8:
        return 3;
    case 4:
        return 2;
    case 2:
        return 1;
    default:
        return 0;
    }
}

uint32_t getOctreeGrid (uint32_t octreeLevel)
{
    switch (octreeLevel)
    {
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

uint32_t getNodeAmount (uint32_t octreeLevel)
{
    switch (octreeLevel)
    {
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

uint32_t getNodeOffset (uint32_t octreeLevel, uint32_t octreeDepth)
{
    uint32_t offset = 0;
    for (uint32_t i = octreeDepth; i > octreeLevel; --i)
    {
        offset += getNodeAmount (i);
    }
    return offset;
}

void create1DKernel (dim3& block, dim3& grid, uint32_t pointCount)
{
    auto blocks = ceil (static_cast<double> (pointCount) / BLOCK_SIZE_MAX);
    auto gridX  = blocks < GRID_SIZE_MAX ? blocks : GRID_SIZE_MAX;
    auto gridY  = ceil (blocks / GRID_SIZE_MAX);

    block = dim3 (BLOCK_SIZE_MAX, 1, 1);
    grid  = dim3 (static_cast<unsigned int> (gridX), static_cast<unsigned int> (gridY), 1);

    printKernelDimensions (block, grid);
}

template <typename coordinateType>
GpuArrayU8 generate_point_cloud_cuboid (uint32_t sideLength, PointCloudInfo& metadata)
{
    coordinateType boundingBoxMax = static_cast<coordinateType> (sideLength) - static_cast<coordinateType> (0.5);
    metadata.pointAmount          = static_cast<uint32_t> (pow (sideLength, 3.0));
    metadata.bbCubic.min          = Vector3<double>{0.5, 0.5, 0.5};
    metadata.bbCubic.max          = Vector3<double>{boundingBoxMax, boundingBoxMax, boundingBoxMax};
    metadata.cloudOffset          = Vector3<double>{0.5, 0.5, 0.5};
    metadata.scale                = {1.0, 1.0, 1.0};
    metadata.pointDataStride      = 12;

    auto pointAmount = sideLength * sideLength * sideLength;
    auto data        = createGpuU8 (pointAmount * 12, "cuboid");

    auto blocks = ceil (pointAmount / 1024.f);
    auto gridX  = blocks < GRID_SIZE_MAX ? blocks : ceil (blocks / GRID_SIZE_MAX);
    auto gridY  = ceil (blocks / gridX);

    dim3 block (BLOCK_SIZE_MAX, 1, 1);
    dim3 grid (static_cast<uint32_t> (gridX), static_cast<uint32_t> (gridY), 1);
    printKernelDimensions (block, grid);

    kernel_point_cloud_cuboid<<<grid, block>>> (data->devicePointer (), pointAmount, sideLength);
    return data;
}

void printKernelDimensions (dim3 block, dim3 grid)
{
    spdlog::debug (
            "Launching kernel with dimensions: "
            "block [{}, {}, {}] | grid[{}, {}, {}]",
            block.x,
            block.y,
            block.z,
            grid.x,
            grid.y,
            grid.z);
}

__host__ __device__ void mapFromDenseIdxToDenseCoordinates (
        Vector3<uint32_t>& coordinates, uint32_t denseVoxelIdx, uint32_t gridSizeLength)
{
    auto xy       = gridSizeLength * gridSizeLength;
    coordinates.z = denseVoxelIdx / xy;
    coordinates.y = (denseVoxelIdx - (coordinates.z * xy)) / gridSizeLength;
    coordinates.x = (denseVoxelIdx - (coordinates.z * xy)) % gridSizeLength;
}

template GpuArrayU8 generate_point_cloud_cuboid<float> (uint32_t sideLength, PointCloudInfo& metadata);

} // namespace tools
