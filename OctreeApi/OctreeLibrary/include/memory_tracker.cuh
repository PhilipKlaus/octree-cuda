/**
 * @file memory_tracker.h
 * @author Philip Klaus
 * @brief Contains a tracker for tracking CUDA memory allocations and de-allocations.
 */
#pragma once

#include "spdlog/spdlog.h"
#include <fstream>

using namespace std::chrono;

constexpr double GB = 1000000000.0;


/**
 * A Memory tracker which tracks CUDA memory allocations and de-allocations.
 * The tracker is globally available and implemented as a singleton.
 * Its main purpose is to show the allocated GPU memory per data structure and to ensure
 * that all of the allocated GPU memory is deallocated in the end.
 */
class MemoryTracker
{
public:
    static MemoryTracker& getInstance ()
    {
        static MemoryTracker instance;
        return instance;
    }

    ~MemoryTracker ()
    {
        std::string data;
        std::string labels;

        for (auto const& event : itsMemAccumulated)
        {
            data += std::to_string (std::get<1> (event));
            labels += ("\"" + std::get<0> (event) + "\"");
            data += ",";
            labels += ",";
        }

        std::ofstream htmlData;
        htmlData.open (std::string (itsMemoryReportFileName), std::ios::out);
        htmlData << itsHtmlPart1;
        htmlData << "{labels: [" << labels
                 << "], datasets: [{label: \"Memory footprint\", backgroundColor: "
                    "\"rgb(255, 99, 132)\", borderColor: \"rgb(255, 99, 132)\",";
        htmlData << "data: [" << data << "]}]},";
        htmlData << itsHtmlPart2;
        htmlData.close ();

        spdlog::info ("Remaining memory on GPU: {:5.5f} [GB]", itsMemoryConsumption);
    }

    void configureMemoryReport (const std::string& filename)
    {
        itsMemoryReportFileName = filename;
    }

    void reservedMemoryEvent (uint64_t memoryFootprint, const std::string& name)
    {
        ++itsMemoryReserveEvents;
        double reservedGB = (memoryFootprint / GB);
        itsCumulatedMemoryReservation += reservedGB;
        itsMemoryConsumption += reservedGB;

        itsMemAccumulated.emplace_back (name, itsMemoryConsumption);
        itsMemPerBuffer.emplace_back (name, reservedGB);

        if (itsMemoryConsumption > itsMemoryPeak)
        {
            itsMemoryPeak = itsMemoryConsumption;
        }
    }

    void freedMemoryEvent (uint64_t memoryFootprint, const std::string& name)
    {
        ++itsMemoryFreeEvents;
        itsMemoryConsumption -= (memoryFootprint / GB);
        itsMemAccumulated.emplace_back ("Deleted " + name, itsMemoryConsumption);
    }

    double getMemoryPeak ()
    {
        return itsMemoryPeak;
    }

    double getCumulatedMemoryReservation ()
    {
        return itsCumulatedMemoryReservation;
    }

    uint32_t getMemoryReserveEvents ()
    {
        return itsMemoryReserveEvents;
    }

    uint32_t getMemoryFreeEvents ()
    {
        return itsMemoryFreeEvents;
    }

    const std::vector<std::tuple<std::string, double>>& getMemoryEvents ()
    {
        return itsMemPerBuffer;
    }

private:
    MemoryTracker ()
    {
        itsMemoryPeak                 = 0;
        itsMemoryConsumption          = 0;
        itsCumulatedMemoryReservation = 0;
        itsMemoryReserveEvents        = 0;
        itsMemoryFreeEvents           = 0;
        itsMemoryReportFileName       = "memory_report.html";
    }

public:
    MemoryTracker (MemoryTracker const&) = delete;
    void operator= (MemoryTracker const&) = delete;

private:
    std::string itsMemoryReportFileName;
    double itsMemoryConsumption;
    double itsMemoryPeak;
    double itsCumulatedMemoryReservation;
    uint32_t itsMemoryReserveEvents;
    uint32_t itsMemoryFreeEvents;

    std::vector<std::tuple<std::string, double>> itsMemAccumulated;
    std::vector<std::tuple<std::string, double>> itsMemPerBuffer;

    const std::string itsHtmlPart1 = "<html>\n"
                                     "    <head>\n"
                                     "        <script src=\"https://cdn.jsdelivr.net/npm/chart.js@2.8.0\"></script>\n"
                                     "        <style>\n"
                                     "            html, body {\n"
                                     "            margin: 0;\n"
                                     "            height: 100%;\n"
                                     "            }\n"
                                     "            canvas {\n"
                                     "            width: 100%;\n"
                                     "            height: 100%;\n"
                                     "            display: block;\n"
                                     "            }\n"
                                     "        </style>\n"
                                     "    </head>\n"
                                     "    <body>\n"
                                     "        <canvas id=\"myChart\"></canvas>\n"
                                     "        <script>\n"
                                     "            var ctx = document.getElementById('myChart').getContext('2d');\n"
                                     "            var chart = new Chart(ctx, {\n"
                                     "                // The type of chart we want to create\n"
                                     "                type: 'line',\n"
                                     "\n"
                                     "                // The data for our dataset\n"
                                     "                data:";

    const std::string itsHtmlPart2 = "                // Configuration options go here\n"
                                     "                options: {\n"
                                     "                    scales: {\n"
                                     "                        yAxes: [{\n"
                                     "                            ticks: {\n"
                                     "                                beginAtZero: true\n"
                                     "                            },\n"
                                     "                            scaleLabel: {\n"
                                     "                               display: true,"
                                     "                               labelString: 'Memory usage [GB]'"
                                     "                            }"
                                     "                        }]\n"
                                     "                    }\n"
                                     "                },\n"
                                     "\n"
                                     "            });\n"
                                     "        </script>\n"
                                     "    </body>\n"
                                     "</html>";
};
