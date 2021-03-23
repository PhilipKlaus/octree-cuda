#include "kernel_helpers.cuh"

__host__ __device__ void mapFromDenseIdxToDenseCoordinates (
        Vector3<uint32_t>& coordinates, uint32_t denseVoxelIdx, uint32_t gridSizeLength)
{
    auto xy       = gridSizeLength * gridSizeLength;
    coordinates.z = denseVoxelIdx / xy;
    coordinates.y = (denseVoxelIdx - (coordinates.z * xy)) / gridSizeLength;
    coordinates.x = (denseVoxelIdx - (coordinates.z * xy)) % gridSizeLength;
}
