//
// Created by KlausP on 20.10.2020.
//

#ifndef OCTREECUDA_EVENTWATCHER_H
#define OCTREECUDA_EVENTWATCHER_H

#include <fstream>
#include "spdlog/spdlog.h"

using namespace std::chrono;

constexpr double GB = 1000000000.0;

const std::string htmlPart1 =
                    "<html>\n"
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
const std::string htmlPart2 =
                    "                // Configuration options go here\n"
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

class EventWatcher {
public:
    static EventWatcher& getInstance()
    {
        static EventWatcher    instance;
        return instance;
    }

    ~EventWatcher() {
        std::string data;
        std::string labels;

        for(auto i = 0; i < itsTimestamps.size(); ++i) {
            data += std::to_string(itsMemoryFootprints[i]);
            labels += ("\"" + itsEventLabels[i] + "\"");
            if(i < itsTimestamps.size() - 1) {
                data += ",";
                labels += ",";
            }
        }

        std::ofstream htmlData;
        htmlData.open (std::string("memory_footprint.html"), std::ios::out);
        htmlData << htmlPart1;
        htmlData << "{labels: [" << labels << "], datasets: [{label: \"Memory footprint\", backgroundColor: \"rgb(255, 99, 132)\", borderColor: \"rgb(255, 99, 132)\",";
        htmlData << "data: [" << data << "]}]},";
        htmlData << htmlPart2;
    }

    void reservedMemoryEvent(uint64_t memoryFootprint, const std::string& name) {
        memoryConsumption += (memoryFootprint / GB);
        recordEvent(name);
    }

    void freedMemoryEvent(uint64_t memoryFootprint, const std::string& name) {
        memoryConsumption -= (memoryFootprint / GB);
        recordEvent("Deleted " + name);
    }

private:
    EventWatcher() {
        itsStart = high_resolution_clock::now();
        memoryConsumption = 0;
    }

    void recordEvent(const std::string& name) {
        itsEventLabels.push_back(name);
        itsMemoryFootprints.push_back(memoryConsumption);
        auto current = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(current - itsStart);
        itsTimestamps.push_back(duration.count());
    }

public:
    EventWatcher(EventWatcher const&)  = delete;
    void operator=(EventWatcher const&) = delete;

private:
    double memoryConsumption;
    std::vector<double> itsMemoryFootprints;
    std::vector<uint64_t> itsTimestamps;
    std::vector<std::string> itsEventLabels;
    time_point<steady_clock> itsStart;
};

#endif //OCTREECUDA_EVENTWATCHER_H
