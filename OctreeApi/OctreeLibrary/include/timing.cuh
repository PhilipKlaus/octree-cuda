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
        cudaEventRecord (itsStart);
    }

    void stop ()
    {
        cudaEventRecord (itsStop);
    }

    float getMilliseconds ()
    {
        float milliseconds = 0;
        cudaEventSynchronize (itsStop);
        cudaEventElapsedTime (&milliseconds, itsStart, itsStop);
        return milliseconds;
    }

private:
    cudaEvent_t itsStart, itsStop;
};

}; // namespace Timing
