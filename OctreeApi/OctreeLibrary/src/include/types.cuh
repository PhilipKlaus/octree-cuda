#pragma once

#include "cuda_array.cuh"
#include <curand_kernel.h>
#include <memory>


struct Node
{
    uint32_t pointCount; // Number of points in the node
    uint32_t parentNode; // The parent node - Only needed during Merging
    bool isFinished;     // Determines if the node is finished (= not mergeable anymore)
    uint64_t dataIdx;    // Determines the position in the data output
    int childNodes[8];   // The 8 child nodes
    bool isInternal;     // Determines if the node is internal (parent) or a leaf
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
using GpuArrayF32     = GpuArray<float>;
using GpuOctree       = GpuArray<Node>;
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
std::unique_ptr<CudaArray<float>> createGpuF32 (Args&&... args)
{
    return createGpu<float> (std::forward<Args> (args)...);
}

template <typename... Args>
std::unique_ptr<CudaArray<uint8_t>> createGpuU8 (Args&&... args)
{
    return createGpu<uint8_t> (std::forward<Args> (args)...);
}

template <typename... Args>
std::unique_ptr<CudaArray<Node>> createGpuOctree (Args&&... args)
{
    return createGpu<Node> (std::forward<Args> (args)...);
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