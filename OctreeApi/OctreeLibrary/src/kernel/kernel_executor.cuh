#pragma once

#include <timing.cuh>
#include <tools.cuh>

// https://developer.nvidia.com/blog/cplusplus-11-in-cuda-variadic-templates/
template <typename... Arguments>
float executeKernel (void (*kernel) (Arguments... args), uint32_t threads, Arguments... args)
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