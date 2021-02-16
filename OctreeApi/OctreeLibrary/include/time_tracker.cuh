/**
 * @file time_tracker.cuh
 * @author Philip Klaus
 * @brief Contains a tracker for runtime measurments
 */

#include <spdlog/spdlog.h>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>


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

    void trackKernelTime (float ms, const std::string& measurement)
    {
        kernelTimings.emplace_back (ms, measurement);
        std::stringstream stream;
        stream << "[kernel] " << measurement << " took: " << ms << " [ms]";
        spdlog::info (stream.str ());
    }

    void trackMemCpyTime (float ms, const std::string& measurement, bool hostToDevice)
    {
        memCopyTimings.emplace_back (ms, measurement);
        std::stringstream stream;
        stream << (hostToDevice ? "[host -> device] " : "[device -> host] ") << measurement << " took: " << ms
               << " [ms]";
        spdlog::info (stream.str ());
    }

    const std::vector<std::tuple<float, std::string>>& getCpuTimings () const
    {
        return cpuTimings;
    }

    const std::vector<std::tuple<float, std::string>>& getKernelTimings () const
    {
        return kernelTimings;
    }

    const std::vector<std::tuple<float, std::string>>& getMemCpyTimings () const
    {
        return memCopyTimings;
    }

private:
    std::vector<std::tuple<float, std::string>> cpuTimings;
    std::vector<std::tuple<float, std::string>> kernelTimings;
    std::vector<std::tuple<float, std::string>> memCopyTimings;
};