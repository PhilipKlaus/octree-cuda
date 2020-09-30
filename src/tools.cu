#include "tools.cuh"
#include <memory>
#include "stdio.h"
#include "types.h"

using namespace std;


__global__ void kernel_point_cloud_cuboid(Point *out, unsigned int n, unsigned int side) {
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

unique_ptr<CudaArray<Point>> generate_point_cloud_cuboid(unsigned int sideLength) {

    auto pointAmount = sideLength * sideLength * sideLength;
    auto data = std::make_unique<CudaArray<Point>>(pointAmount);

    auto blocks = ceil(pointAmount / 1024.f);
    auto gridX = blocks < GRID_SIZE_MAX ? blocks : ceil(blocks / GRID_SIZE_MAX);
    auto gridY = ceil(blocks / gridX);

    dim3 block(BLOCK_SIZE_MAX, 1, 1);
    dim3 grid(static_cast<unsigned int>(gridX), static_cast<unsigned int>(gridY), 1);

    cout << "Launching kernle with threads per block: " << block.x << ", blocks: " << blocks << ", gridX: "
    << gridX << ", gridX: " << gridY << endl;

    kernel_point_cloud_cuboid <<<  grid, block >>> (data->rawPointer(), pointAmount, sideLength);
    return data;
}