#include <iostream>

#include "tools.cuh"
#include "chunking.cuh"

using namespace std;

constexpr unsigned int GRID_SIZE = 128;

int main() {

    // Create equally spaced point cloud cuboid
    unsigned int elementsPerCuboidSide = 128;
    unique_ptr<CudaArray<Point>> data = generate_point_cloud_cuboid(elementsPerCuboidSide);

    // Perform chunking _> build the global octree hierarchy
    Vector3 posOffset = {0.5, 0.5 , 0.5};
    Vector3 size = {127, 127, 127};
    Vector3 minimum = {0.5, 0.5, 0.5};

    auto countingGrid = initialPointCounting(move(data), GRID_SIZE, posOffset, size, minimum);
}