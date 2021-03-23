#pragma once

#include "cuda_array.cuh"
#include <curand_kernel.h>
#include <memory>


struct Chunk
{
    uint32_t pointCount;       // How many points does this chunk have
    uint32_t parentChunkIndex; // Determines the INDEX of the parent CHUNK in the GRID - Only needed during Merging
    bool isFinished;           // Is this chunk finished (= not mergeable anymore)
    uint64_t chunkDataIndex;   // Determines the INDEX in the chunk data array -> for storing point data
    int childrenChunks[8];     // The INDICES of the children chunks in the GRID
    bool isParent;             // Denotes if Chunk is a parent or a leaf node
};

#pragma pack(push, 1)
struct OutputBuffer
{
    int32_t x, y, z;
    uint16_t r, g, b;
};
#pragma pack(pop)

using PointLut = uint32_t;

template <typename gpuType>
using GpuArray        = std::unique_ptr<CudaArray<gpuType>>;
using GpuArrayU8      = GpuArray<uint8_t>;
using GpuArrayU32     = GpuArray<uint32_t>;
using GpuArrayI32     = GpuArray<int>;
using GpuOctree       = GpuArray<Chunk>;
using GpuAveraging    = GpuArray<uint64_t>;
using GpuRandomState  = GpuArray<curandState_t>;
using GpuPointLut     = GpuArray<PointLut>;
using GpuOutputBuffer = GpuArray<OutputBuffer>;

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
std::unique_ptr<CudaArray<curandState_t>> createGpuRandom (Args&&... args)
{
    return createGpu<curandState_t> (std::forward<Args> (args)...);
}

template <typename... Args>
std::unique_ptr<CudaArray<uint64_t>> createGpuAveraging (Args&&... args)
{
    return createGpu<uint64_t> (std::forward<Args> (args)...);
}

template <typename... Args>
std::unique_ptr<CudaArray<PointLut>> createGpuOutputData (Args&&... args)
{
    return createGpu<PointLut> (std::forward<Args> (args)...);
}

template <typename... Args>
std::unique_ptr<CudaArray<OutputBuffer>> createGpuOutputBuffer (Args&&... args)
{
    return createGpu<OutputBuffer> (std::forward<Args> (args)...);
}