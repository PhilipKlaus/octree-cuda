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
#include "eventWatcher.h"

using namespace std;

struct Vector3
{
    float x, y, z;
};

struct Chunk {
    uint64_t pointCount;
    Chunk *dst;
    bool isFinished;
    uint64_t indexCount;
    uint64_t treeIndex;
};

struct BoundingBox {
    Vector3 minimum;
    Vector3 maximum;
};

struct PointCloudMetadata {
    uint64_t pointAmount;
    BoundingBox boundingBox;
    Vector3 cloudOffset;
    Vector3 scale;
};

template <typename dataType>
class CudaArray {

public:

    CudaArray(uint64_t elements, const std::string& name) :
    itsElements(elements),
    itsName(name) {
        auto memoryToReserve = itsElements * sizeof(dataType);
        itsMemory = memoryToReserve;
        itsWatcher.reservedMemoryEvent(memoryToReserve, itsName);
        cudaMalloc((void**)&itsData, memoryToReserve);
        spdlog::debug("Reserved memory on GPU for {} elements with a size of {} bytes", elements, memoryToReserve);
    }

    ~CudaArray() {
        itsWatcher.freedMemoryEvent(itsMemory, itsName);
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

    uint64_t pointCount() {
        return itsElements;
    }
private:
    std::string itsName;
    uint64_t itsMemory;
    uint64_t itsElements;
    dataType *itsData;
    EventWatcher& itsWatcher = EventWatcher::getInstance();
};

#endif