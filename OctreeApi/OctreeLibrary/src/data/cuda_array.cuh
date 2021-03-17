//
// Created by KlausP on 28.10.2020.
//

#pragma once

#include <memory>
#include <string>

#include "defines.cuh"
#include "memory_tracker.cuh"
#include "time_tracker.cuh"

using namespace std;

template <typename dataType>
class CudaArray
{
public:
    CudaArray (uint64_t elements, const std::string& name) : itsElements (elements), itsName (name)
    {
        auto memoryToReserve = itsElements * sizeof (dataType);
        itsMemory            = memoryToReserve;
        itsWatcher.reservedMemoryEvent (memoryToReserve, itsName);

        auto start = std::chrono::high_resolution_clock::now ();
        gpuErrchk (cudaMalloc ((void**)&itsData, memoryToReserve));
        auto stop                             = std::chrono::high_resolution_clock::now ();
        std::chrono::duration<double> elapsed = stop - start;
        Timing::TimeTracker::getInstance ().trackMemAllocTime (
                static_cast<float> (elapsed.count () * 1000), itsName, false);

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

        auto start = std::chrono::high_resolution_clock::now ();
        gpuErrchk (cudaMemcpy (host.get (), itsData, sizeof (dataType) * itsElements, cudaMemcpyDeviceToHost));
        auto stop                             = std::chrono::high_resolution_clock::now ();
        std::chrono::duration<double> elapsed = stop - start;
        Timing::TimeTracker::getInstance ().trackMemCpyTime (
                static_cast<float> (elapsed.count () * 1000), itsName, false);

        return host;
    }

    void toGPU (uint8_t* host)
    {
        auto start = std::chrono::high_resolution_clock::now ();
        gpuErrchk (cudaMemcpy (itsData, host, sizeof (dataType) * itsElements, cudaMemcpyHostToDevice));
        auto stop                             = std::chrono::high_resolution_clock::now ();
        std::chrono::duration<double> elapsed = stop - start;
        Timing::TimeTracker::getInstance ().trackMemCpyTime (
                static_cast<float> (elapsed.count ()) * 1000, itsName, false);
    }

    uint64_t pointCount () const
    {
        return itsElements;
    }

    void memset (int value)
    {
        gpuErrchk (cudaMemset (itsData, value, itsElements * sizeof (dataType)));
    }

    static unique_ptr<CudaArray<dataType>> fromDevicePtr (dataType* device, uint32_t elements, const std::string& name)
    {
        return std::make_unique<CudaArray<dataType>> (device, elements, name);
    }

public:
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
    uint64_t itsElements;
    dataType* itsData;
    MemoryTracker& itsWatcher = MemoryTracker::getInstance ();
};
