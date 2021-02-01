#pragma once

#include "timing.cuh"
#include "tools.cuh"

#include "subsample_evaluating.cuh"

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


namespace Kernel {

struct KernelConfig
{
    CloudType cloudType;
    uint32_t threadAmount;
};


template <typename... Arguments>
float evaluateSubsamples (KernelConfig config, Arguments&&... args)
{
    if (config.cloudType == CLOUD_FLOAT_UINT8_T)
    {
        return executeKernel (
                subsampling::kernelEvaluateSubsamples<float>, config.threadAmount, std::forward<Arguments> (args)...);
    }
    else
    {
        return executeKernel (
                subsampling::kernelEvaluateSubsamples<double>, config.threadAmount, std::forward<Arguments> (args)...);
    }
}

} // namespace Kernel
