#pragma once

#include "timing.cuh"
#include "tools.cuh"


// https://developer.nvidia.com/blog/cplusplus-11-in-cuda-variadic-templates/
template <typename FunctType, typename... Arguments>
void executeKernel (FunctType kernel, uint32_t threads, const std::string& name, Arguments&&... args)
{
    // Calculate kernel dimensions
    dim3 grid, block;
    tools::create1DKernel (block, grid, threads);

#ifdef KERNEL_TIMINGS
    Timing::KernelTimer timer;
    timer.start ();
#endif
    kernel<<<grid, block>>> (std::forward<Arguments> (args)...);
#ifdef KERNEL_TIMINGS
    timer.stop ();
    Timing::TimeTracker::getInstance ().trackKernelTime (timer, name);
#endif
#ifdef ERROR_CHECKS
    cudaDeviceSynchronize ();
#endif
    gpuErrchk (cudaGetLastError ());
}


namespace Kernel {

struct KernelConfig
{
    CloudType cloudType;
    uint32_t threadAmount;
    std::string name;
};


} // namespace Kernel
