/**
 * @file time_tracker.cuh
 * @author Philip Klaus
 * @brief Contains a tracker for runtime measurments
 */

#pragma once
#include "timing.cuh"
#include <map>
#include <set>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace Timing {

struct TimingProps
{
    float duration;
    uint32_t invocations;
};

enum Time
{
    PROCESS,
    MEM_CPY,
    MEM_ALLOC
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
        if (kernelOrder.find (measurement) == kernelOrder.end ())
        {
            kernelOrder[measurement] = static_cast<int> (kernelTimings.size ());
            kernelTimings.emplace_back (measurement, TimingProps{0.f, 0});
        }
    }

    void trackMemCpyTime (double ms, const std::string& measurement, bool log)
    {
        memCopyTimings.emplace_back (ms, measurement);
        if (log)
        {
            spdlog::info ("{:<15} {:<30} took: {} [ms]", "[memcpy]", measurement, ms);
        }
    }

    void trackMemAllocTime (double ms, const std::string& measurement, bool log)
    {
        memAllocTimings.emplace_back (ms, measurement);
        if (log)
        {
            spdlog::info ("{:<15} {:<30} took: {} [ms]", "[cudamalloc]", measurement, ms);
        }
    }

    void trackProcessTime (double ms, const std::string& measurement, bool log)
    {
        processTimings.emplace_back (ms, measurement);
        if (log)
        {
            spdlog::info ("{:<15} {:<30} took: {} [ms]", "[process]", measurement, ms);
        }
    }

    const std::vector<std::tuple<std::string, TimingProps>>& getKernelTimings ()
    {
        if (!kernelTimers.empty ())
        {
            spdlog::info ("--------------------------------");
            for (auto& timer : kernelTimers)
            {
                float ms  = std::get<0> (timer).getDuration ();
                int order = kernelOrder[std::get<1> (timer)];
                std::get<1> (kernelTimings[order]).duration += ms;
                std::get<1> (kernelTimings[order]).invocations += 1;
            }
            float accKernelDur = 0.f;
            uint32_t accKernelInv = 0;
            for (auto& timing : kernelTimings)
            {
                float dur    = std::get<1> (timing).duration;
                accKernelDur += dur;
                uint32_t inv = std::get<1> (timing).invocations;
                accKernelInv += inv;
                spdlog::info ("[kernel] {:<30} invocations: {} took: {} [ms]", std::get<0> (timing), inv, dur);
            }
            spdlog::info ("--------------------------------");
            for (auto& timing : kernelTimings)
            {
                float dur = std::get<1> (timing).duration;
                spdlog::info ("[kernel] {:<30} took: {} [%]", std::get<0> (timing), (dur/accKernelDur) * 100.f);
            }
            spdlog::info ("--------------------------------");
            spdlog::info (
                    "[KERNEL-SUMMARY] invocations: {} accumulated duration: {} [ms]",
                    accKernelInv,
                    accKernelDur);
        }
        return kernelTimings;
    }

    const std::vector<std::tuple<double, std::string>>& getMemCpyTimings () const
    {
        return memCopyTimings;
    }

    const std::vector<std::tuple<double, std::string>>& getMemAllocTimings () const
    {
        return memAllocTimings;
    }

    const std::vector<std::tuple<double, std::string>>& getProcessTimings () const
    {
        return processTimings;
    }

    static time_point<steady_clock> start ()
    {
        return std::chrono::high_resolution_clock::now ();
    }

    static double stop (
            const time_point<steady_clock>& start, const std::string& measurement, Time kind, bool log = true)
    {
        auto finish                           = std::chrono::high_resolution_clock::now ();
        std::chrono::duration<double> elapsed = finish - start;
        double ms                             = elapsed.count () * 1000;

        switch (kind)
        {
        case Time::PROCESS:
            getInstance ().trackProcessTime (ms, measurement, log);
            break;
        case Time::MEM_ALLOC:
            getInstance ().trackMemAllocTime (ms, measurement, log);
            break;
        default:
            getInstance ().trackMemCpyTime (ms, measurement, log);
            break;
        }
        return ms;
    }

private:
    std::vector<std::tuple<double, std::string>> memCopyTimings;
    std::vector<std::tuple<double, std::string>> memAllocTimings;
    std::vector<std::tuple<double, std::string>> processTimings;

    // Kernel timings
    std::vector<std::tuple<KernelTimer, std::string>> kernelTimers;
    std::vector<std::tuple<std::string, TimingProps>> kernelTimings;
    std::map<std::string, int> kernelOrder;
};
} // namespace Timing
