//
// Created by KlausP on 30.09.2020.
//

#ifndef OCTREECUDA_TYPES
#define OCTREECUDA_TYPES

#include <memory>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include "spdlog/spdlog.h"

using namespace std;

struct Vector3
{
    float x, y, z;
};

struct Chunk {
    uint32_t count;
    Chunk *dst;
    bool isFinished;
    uint32_t indexCount;
    uint32_t treeIndex;
};

struct BoundingBox {
    Vector3 minimum;
    Vector3 maximum;

    Vector3 size() const {
        return {
            maximum.x - minimum.x,
            maximum.y - minimum.y,
            maximum.z - minimum.z
        };
    }
};

struct PointCloudMetadata {
    uint32_t pointAmount;
    BoundingBox boundingBox;
    Vector3 cloudOffset;
};

template <typename dataType>
class CudaArray {

public:

    CudaArray(unsigned int elements) : itsElements(elements) {
        auto memoryToReserve = itsElements * sizeof(dataType);
        cudaMalloc((void**)&itsData, memoryToReserve);
        spdlog::debug("Reserved memory on GPU for {} elements with a size of {} bytes", elements, memoryToReserve);
    }

    ~CudaArray() {
        cudaFree(itsData);
        spdlog::debug("Freed GPU memory: {} bytes", itsElements * sizeof(dataType));
    }

    dataType* devicePointer() {
        return itsData;
    }

    unique_ptr<dataType[]> toHost() {
        unique_ptr<dataType[]> host (new dataType[itsElements]);
        cudaMemcpy(host.get(), itsData, sizeof(dataType) * itsElements, cudaMemcpyDeviceToHost);
        return host;
    }

    void toGPU(uint8_t *host) {
        cudaMemcpy(itsData, host, sizeof(dataType) * itsElements, cudaMemcpyHostToDevice);
    }

    uint32_t pointCount() {
        return itsElements;
    }

    unsigned int itsElements;
    dataType *itsData;
};

#endif