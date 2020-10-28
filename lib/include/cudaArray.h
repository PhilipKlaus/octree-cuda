//
// Created by KlausP on 28.10.2020.
//

#ifndef OCTREECUDA_CUDAARRAY_H
#define OCTREECUDA_CUDAARRAY_H

#include <cstdint>
#include <string>
#include <memory>
#include "eventWatcher.h"
#include "../src/defines.cuh"

template <typename dataType>
class CudaArray {

public:

    CudaArray(uint32_t elements, const std::string& name) :
            itsElements(elements),
            itsName(name) {
        auto memoryToReserve = itsElements * sizeof(dataType);
        itsMemory = memoryToReserve;
        itsWatcher.reservedMemoryEvent(memoryToReserve, itsName);
        gpuErrchk(cudaMalloc((void**)&itsData, memoryToReserve));
        spdlog::debug("Reserved memory on GPU for {} elements with a size of {} bytes", elements, memoryToReserve);
    }

    ~CudaArray() {
        itsWatcher.freedMemoryEvent(itsMemory, itsName);
        gpuErrchk(cudaFree(itsData));
        spdlog::debug("Freed GPU memory: {} bytes", itsElements * sizeof(dataType));
    }

    dataType* devicePointer() {
        return itsData;
    }

    std::unique_ptr<dataType[]> toHost() {
        unique_ptr<dataType[]> host (new dataType[itsElements]);
        gpuErrchk(cudaMemcpy(host.get(), itsData, sizeof(dataType) * itsElements, cudaMemcpyDeviceToHost));
        return host;
    }

    void toGPU(uint8_t *host) {
        gpuErrchk(cudaMemcpy(itsData, host, sizeof(dataType) * itsElements, cudaMemcpyHostToDevice));
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


#endif //OCTREECUDA_CUDAARRAY_H
