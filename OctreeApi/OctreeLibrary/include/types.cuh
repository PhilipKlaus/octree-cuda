#pragma once

#include <curand_kernel.h>
#include <memory>

#include "../src/include/cudaArray.h"


struct Averaging
{
    uint32_t r, g, b;
    uint32_t pointCount;
};

struct SubsampleConfig
{
    uint32_t* lutAdress;
    Averaging* averagingAdress;
    uint32_t lutStartIndex;
    uint32_t pointOffsetLower;
    uint32_t pointOffsetUpper;
};

struct __align__ (16) SubsampleSet
{
    SubsampleConfig child_0;
    SubsampleConfig child_1;
    SubsampleConfig child_2;
    SubsampleConfig child_3;
    SubsampleConfig child_4;
    SubsampleConfig child_5;
    SubsampleConfig child_6;
    SubsampleConfig child_7;
};

struct Chunk
{
    uint32_t pointCount;       // How many points does this chunk have
    uint32_t parentChunkIndex; // Determines the INDEX of the parent CHUNK in the GRID - Only needed during Merging
    bool isFinished;           // Is this chunk finished (= not mergeable anymore)
    uint32_t chunkDataIndex;   // Determines the INDEX in the chunk data array -> for storing point data
    int childrenChunks[8];     // The INDICES of the children chunks in the GRID
    bool isParent;             // Denotes if Chunk is a parent or a leaf node
};

struct SubsamplingTimings
{
    float subsampleEvaluation;
    float generateRandoms;
    float averaging;
    float subsampling;
};


template <typename gpuType>
using GpuArray       = std::unique_ptr<CudaArray<gpuType>>;
using GpuArrayU8     = GpuArray<uint8_t>;
using GpuArrayU32    = GpuArray<uint32_t>;
using GpuArrayI32    = GpuArray<int>;
using GpuOctree      = GpuArray<Chunk>;
using GpuSubsample   = GpuArray<SubsampleConfig>;
using GpuAveraging   = GpuArray<Averaging>;
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