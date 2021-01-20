#pragma once

#include "../include/tools.cuh"
#include <timing.cuh>

// https://developer.nvidia.com/blog/cplusplus-11-in-cuda-variadic-templates/
template <typename FunctType, typename... Arguments>
float executeKernel (FunctType kernel, uint32_t threads, Arguments&&... args)
{
    // Calculate kernel dimensions
    dim3 grid, block;
    tools::create1DKernel (block, grid, threads);

    tools::KernelTimer timer;
    timer.start ();
    kernel<<<grid, block>>> (std::forward<Arguments> (args)...);
    timer.stop ();
    gpuErrchk (cudaGetLastError ());
    return timer.getMilliseconds ();
}