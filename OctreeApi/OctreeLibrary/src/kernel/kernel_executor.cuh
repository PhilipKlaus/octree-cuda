#pragma once

#include "timing.cuh"
#include "tools.cuh"


// https://developer.nvidia.com/blog/cplusplus-11-in-cuda-variadic-templates/
template <typename FunctType, typename... Arguments>
float executeKernel (FunctType kernel, uint32_t threads, Arguments&&... args)
{
    // Calculate kernel dimensions
    dim3 grid, block;
    tools::create1DKernel (block, grid, threads);
#ifdef CUDA_TIMINGS
    tools::KernelTimer timer;
    timer.start ();
    kernel<<<grid, block>>> (std::forward<Arguments> (args)...);
    timer.stop ();
    gpuErrchk (cudaGetLastError ());
    return timer.getMilliseconds ();
#else
    kernel<<<grid, block>>> (std::forward<Arguments> (args)...);
    gpuErrchk (cudaGetLastError ());
    return 0.f;
#endif
}


namespace Kernel {

struct KernelConfig
{
    CloudType cloudType;
    uint32_t threadAmount;
};


} // namespace Kernel
