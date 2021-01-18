#pragma once

#include <tools.cuh>

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
