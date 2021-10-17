#pragma once

#include <cuda_runtime_api.h>


namespace Timing {

class KernelTimer
{
public:
    KernelTimer ()
    {
        cudaEventCreate (&itsStart);
        cudaEventCreate (&itsStop);
    }

    void start ()
    {
        cudaDeviceSynchronize ();
        cudaEventRecord (itsStart);
    }

    void stop ()
    {
        cudaEventRecord (itsStop);
        cudaEventSynchronize (itsStop);
        cudaDeviceSynchronize ();
        cudaEventElapsedTime (&milliseconds, itsStart, itsStop);
    }

    float getDuration ()
    {
        return milliseconds;
    }

private:
    cudaEvent_t itsStart, itsStop;
    float milliseconds = 0.0;
};

}; // namespace Timing
