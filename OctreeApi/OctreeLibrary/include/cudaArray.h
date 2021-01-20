//
// Created by KlausP on 28.10.2020.
//

#pragma once

#include "../src/misc/defines.cuh"
#include "eventWatcher.h"
#include <memory>
#include <string>

using namespace std;

template <typename dataType>
class CudaArray
{
public:
    CudaArray (uint32_t elements, const std::string& name) : itsElements (elements), itsName (name)
    {
        auto memoryToReserve = itsElements * sizeof (dataType);
        itsMemory            = memoryToReserve;
        itsWatcher.reservedMemoryEvent (memoryToReserve, itsName);
        gpuErrchk (cudaMalloc ((void**)&itsData, memoryToReserve));
        spdlog::debug ("Reserved GPU memory: {} bytes, {} elements", elements, memoryToReserve);
    }

    ~CudaArray ()
    {
        itsWatcher.freedMemoryEvent (itsMemory, itsName);
        gpuErrchk (cudaFree (itsData));
        spdlog::debug ("Freed GPU memory: {} bytes", itsElements * sizeof (dataType));
    }

    dataType* devicePointer () const
    {
        return itsData;
    }

    std::unique_ptr<dataType[]> toHost () const
    {
        unique_ptr<dataType[]> host (new dataType[itsElements]);
        gpuErrchk (cudaMemcpy (host.get (), itsData, sizeof (dataType) * itsElements, cudaMemcpyDeviceToHost));
        return host;
    }

    void toGPU (uint8_t* host)
    {
        gpuErrchk (cudaMemcpy (itsData, host, sizeof (dataType) * itsElements, cudaMemcpyHostToDevice));
    }

    uint32_t pointCount () const
    {
        return itsElements;
    }

    void memset (dataType value)
    {
        gpuErrchk (cudaMemset (itsData, value, itsElements * sizeof (dataType)));
    }

    static unique_ptr<CudaArray<dataType>> fromDevicePtr (dataType* device, uint32_t elements, const std::string& name)
    {
        return std::make_unique<CudaArray<dataType>> (device, elements, name);
    }

private:
    explicit CudaArray (dataType* device, uint32_t elements, const std::string& name) :
            itsElements (elements), itsName (name), itsData (device)
    {
        auto memoryToReserve = itsElements * sizeof (dataType);
        itsMemory            = memoryToReserve;
        spdlog::debug ("Importe GPU memory: {} bytes, {} elements", elements, memoryToReserve);
    }

private:
    std::string itsName;
    uint64_t itsMemory;
    uint32_t itsElements;
    dataType* itsData;
    EventWatcher& itsWatcher = EventWatcher::getInstance ();
};
