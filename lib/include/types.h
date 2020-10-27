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
    uint32_t pointCount;        // How many points does this chunk have
    uint32_t parentChunkIndex;  // Determines the INDEX of the parent CHUNK in the GRID - Only needed during Merging
    bool isFinished;            // Is this chunk finished (= not mergeable anymore)
    uint32_t chunkDataIndex;    // Determines the INDEX in the chunk data array -> for storing point data
    uint32_t childrenChunks[8]; // The INDICES of the children chunks in the GRID
};

struct BoundingBox {
    Vector3 minimum;
    Vector3 maximum;
};

struct PointCloudMetadata {
    uint32_t pointAmount;
    BoundingBox boundingBox;
    Vector3 cloudOffset;
    Vector3 scale;
};

template <typename dataType>
class CudaArray {

public:

    CudaArray(uint32_t elements, const std::string& name) :
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

    uint32_t pointCount() {
        return itsElements;
    }

private:
    std::string itsName;
    uint64_t itsMemory;
    uint32_t itsElements;
    dataType *itsData;
    EventWatcher& itsWatcher = EventWatcher::getInstance();
};

#endif