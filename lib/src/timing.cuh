#ifndef OCTREE_TIMING
#define OCTREE_TIMING

#include <cuda.h>

namespace tools {

    class KernelTimer {

    public:
        KernelTimer() {
            cudaEventCreate(&itsStart);
            cudaEventCreate(&itsStop);
        }

        void start() {
            cudaEventRecord(itsStart);
        }

        void stop() {
            cudaEventRecord(itsStop);
            cudaEventSynchronize(itsStop);
        }

        float getMilliseconds() {
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, itsStart, itsStop);
            return milliseconds;
        }

    private:
        cudaEvent_t itsStart, itsStop;
    };

};

#endif