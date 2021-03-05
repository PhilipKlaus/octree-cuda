/**
 * @file time_tracker.cuh
 * @author Philip Klaus
 * @brief Contains a tracker for runtime measurments
 */

#pragma once
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include "timing.cuh"

namespace Timing {

/**
 * A tracker for montoring runtime measurements.
 * The tracker differs between CPU, GPU and Memory-related time measurements and
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

    void trackCpuTime (float ms, const std::string& measurement)
    {
        cpuTimings.emplace_back (ms, measurement);
        std::stringstream stream;
        stream << "[cpu task] " << measurement << " took: " << ms << " [ms]";
        spdlog::info (stream.str ());
    }

    void trackKernelTime (KernelTimer timer, const std::string& measurement)
    {
        kernelTimings.emplace_back (timer, measurement);
        //std::stringstream stream;
        //stream << "[kernel] " << measurement << " took: " << ms << " [ms]";
        //spdlog::info (stream.str ());
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

    const std::vector<std::tuple<float, std::string>>& getCpuTimings () const
    {
        return cpuTimings;
    }

    const std::vector<std::tuple<float, std::string>>& getKernelTimings () const
    {
        spdlog::info("----- KERNEL TIMINGS -----");
        std::vector<std::tuple<float, std::string>> timings;
        for (auto entry: kernelTimings) {
            float ms = std::get<0>(entry).getMilliseconds();
            timings.emplace_back (ms, std::get<1>(entry));
            spdlog::info("[kernel] {} took: {} [ms]", std::get<1>(entry), ms);
        }
        return timings;
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
    std::vector<std::tuple<float, std::string>> cpuTimings;
    std::vector<std::tuple<KernelTimer, std::string>> kernelTimings;
    std::vector<std::tuple<float, std::string>> memCopyTimings;
    std::vector<std::tuple<float, std::string>> memAllocTimings;
};
}
