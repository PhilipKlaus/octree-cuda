#pragma once

#include <curand_kernel.h>
#include <memory>

template <typename gpuType>
using GpuArray      = unique_ptr<CudaArray<gpuType>>;
using GpuArrayU8    = GpuArray<uint8_t>;
using GpuArrayU32   = GpuArray<uint32_t>;
using GpuArrayI32   = GpuArray<int>;
using GpuOctree     = GpuArray<Chunk>;
using GpuSubsample  = GpuArray<SubsampleConfig>;
using GpuAveraging  = GpuArray<Averaging>;
using GpuRandomState = GpuArray<curandState_t>;

template <typename T, typename... Args>
std::unique_ptr<CudaArray<T>> createGpu (Args&&... args)
{
    return std::make_unique<CudaArray<T>> (std::forward<Args> (args)...);
}

template <typename... Args>
std::unique_ptr<CudaArray<uint32_t>> createGpuU32 (Args&&... args)
{
    return createGpu<uint32_t> (std::forward<Args> (args)...);
}

template <typename... Args>
std::unique_ptr<CudaArray<int>> createGpuI32 (Args&&... args)
{
    return createGpu<int> (std::forward<Args> (args)...);
}

template <typename... Args>
std::unique_ptr<CudaArray<uint8_t>> createGpuU8 (Args&&... args)
{
    return createGpu<uint8_t> (std::forward<Args> (args)...);
}

template <typename... Args>
std::unique_ptr<CudaArray<Chunk>> createGpuOctree (Args&&... args)
{
    return createGpu<Chunk> (std::forward<Args> (args)...);
}

template <typename... Args>
std::unique_ptr<CudaArray<SubsampleConfig>> createGpuSubsample (Args&&... args)
{
    return createGpu<SubsampleConfig> (std::forward<Args> (args)...);
}

template <typename... Args>
std::unique_ptr<CudaArray<curandState_t>> createGpuRandom (Args&&... args)
{
    return createGpu<curandState_t> (std::forward<Args> (args)...);
}

template <typename... Args>
std::unique_ptr<CudaArray<Averaging>> createGpuAveraging (Args&&... args)
{
    return createGpu<Averaging> (std::forward<Args> (args)...);
}