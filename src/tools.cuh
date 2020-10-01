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


template <typename dataType>
class CudaArray {

public:

    CudaArray(unsigned int elements) : itsElements(elements) {
        cudaMalloc((void**)&itsData, itsElements * sizeof(dataType));
        cout << "Reserved memory on GPU" << " for " << elements << " pointCount ["
             << sizeof(dataType) * elements << " bytes] " << endl;
    }

    ~CudaArray() {
        cudaFree(itsData);
        cout << "Freed memory on GPU" << " from " << itsElements << " pointCount ["
             << sizeof(dataType) * itsElements << " bytes] " << endl;
    }

    dataType* devicePointer() {
        return itsData;
    }

    unique_ptr<dataType[]> toHost() {
        unique_ptr<dataType[]> host (new dataType[itsElements]);
        cudaMemcpy(host.get(), itsData, sizeof(dataType) * itsElements, cudaMemcpyDeviceToHost);
        return host;
    }

    uint32_t pointCount() {
        return itsElements;
    }

    unsigned int itsElements;
    dataType *itsData;
};

unique_ptr<CudaArray<Point>> generate_point_cloud_cuboid(unsigned int sideLength);

static inline unsigned int divUp (const int64_t a, const int64_t b)
{
    return static_cast<unsigned int> ((a % b != 0) ? (a / b + 1) : (a / b));
}

void createThreadPerPointKernel(dim3 &block, dim3 &grid, uint32_t pointCount);

#endif