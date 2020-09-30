#ifndef OCTREECUDA_TOOLS
#define OCTREECUDA_TOOL

#include <cuda_runtime_api.h>
#include <cuda.h>
# include <iostream>
# include <memory>
#include "types.h"

using namespace std;

constexpr unsigned int BLOCK_SIZE_MAX = 1024;
constexpr unsigned int GRID_SIZE_MAX = 65535;

template <typename dataType>
class CudaArray {

public:
    CudaArray(unsigned int elements) {
        itsElements = elements;
        cudaMalloc((void**)&itsData, elements * sizeof(dataType));

        cout << "Reserved memory on GPU " << " with " << elements << " ["
            << sizeof(dataType) * elements << " bytes] " << endl;
    }

    ~CudaArray() {
        cudaFree(itsData);
        cout << "Freed memory on GPU";
    }

    dataType* rawPointer() {
        return itsData;
    }

    unique_ptr<dataType[]> toHost() {
        unique_ptr<dataType[]> host (new dataType[itsElements]);
        cudaMemcpy(host.get(), itsData, sizeof(dataType) * itsElements, cudaMemcpyDeviceToHost);
        return host;
    }

private:
    unsigned int itsElements;
    dataType *itsData;
    dataType *itsHostPointer;
    dataType *itsDevicePointer;
};

unique_ptr<CudaArray<Point>> generate_point_cloud_cuboid(unsigned int sideLength);

static inline unsigned int divUp (const int64_t a, const int64_t b)
{
    return static_cast<unsigned int> ((a % b != 0) ? (a / b + 1) : (a / b));
}

# endif