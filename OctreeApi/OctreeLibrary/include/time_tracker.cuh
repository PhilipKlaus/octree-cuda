/**
 * @file time_tracker.cuh
 * @author Philip Klaus
 * @brief Contains a tracker for runtime measurments
 */

#pragma once
#include "timing.cuh"
#include <set>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <map>

namespace Timing {

struct TimingProps {
    float duration;
    uint32_t invocations;
};

/**
 * A tracker for montoring gpu related runtimes.
 * The tracker differs between Cuda kernel and Cuda memory related time measurements and
 * is implemented as a singleton.
 */
class TimeTracker
{
public:
    TimeTracker ()                   = default;
    TimeTracker (TimeTracker const&) = delete;
    void operator= (TimeTracker const&) = delete;

    static TimeTracker& getInstance ()
    {
        static TimeTracker instance;
        return instance;
    }

    void trackKernelTime (KernelTimer timer, const std::string& measurement)
    {
        kernelTimers.emplace_back (timer, measurement);
        if(kernelOrder.find(measurement) == kernelOrder.end()) {
            kernelOrder[measurement] = kernelTimings.size();
            kernelTimings.emplace_back(measurement, TimingProps{0.f, 0});
        }
    }

    void trackMemCpyTime (float ms, const std::string& measurement, bool hostToDevice, bool silent = true)
    {
        memCopyTimings.emplace_back (ms, measurement);
        if (!silent)
        {
            std::stringstream stream;
            stream << (hostToDevice ? "[host -> device] " : "[device -> host] ") << measurement << " took: " << ms
                   << " [ms]";
            spdlog::info (stream.str ());
        }
    }

    void trackMemAllocTime (float ms, const std::string& measurement, bool silent = true)
    {
        memAllocTimings.emplace_back (ms, measurement);
        if (!silent)
        {
            std::stringstream stream;
            stream << "[cudaMalloc] for '" << measurement << "' took: " << ms << " [ms]";
            spdlog::info (stream.str ());
        }
    }

    const std::vector<std::tuple<std::string, TimingProps>>& getKernelTimings ()
    {
        spdlog::info("----- KERNEL TIMINGS -----");
        for (auto &timer: kernelTimers) {
            float ms = std::get<0>(timer).getMilliseconds();
            int order = kernelOrder[std::get<1>(timer)];
            std::get<1>(kernelTimings[order]).duration += ms;
            std::get<1>(kernelTimings[order]).invocations += 1;
        }
        for(auto &timing: kernelTimings) {
            float dur = std::get<1>(timing).duration;
            uint32_t inv = std::get<1>(timing).invocations;
            spdlog::info("[kernel] {:<30} invocations: {} took: {} [ms]", std::get<0>(timing), inv, dur);
        }
        return kernelTimings;
    }

    const std::vector<std::tuple<float, std::string>>& getMemCpyTimings () const
    {
        return memCopyTimings;
    }

    const std::vector<std::tuple<float, std::string>>& getMemAllocTimings () const
    {
        return memAllocTimings;
    }

private:
    std::vector<std::tuple<float, std::string>> memCopyTimings;
    std::vector<std::tuple<float, std::string>> memAllocTimings;

    std::vector<std::tuple<KernelTimer, std::string>> kernelTimers;
    std::vector<std::tuple<std::string, TimingProps>> kernelTimings;
    std::map<std::string, int> kernelOrder;
};
}
