//
// Created by KlausP on 20.10.2020.
//

#ifndef OCTREECUDA_EVENTWATCHER_H
#define OCTREECUDA_EVENTWATCHER_H

#include <fstream>
#include "spdlog/spdlog.h"

using namespace std::chrono;

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
        std::ofstream htmlData;
        htmlData.open (std::string("memory_footprint.html"), std::ios::out);
        htmlData << htmlPart1;
        htmlData << R"lit({labels: ["0","1","2","3","4","5","6","7","8","9","10"], datasets: [{label: "Memory footprint", backgroundColor: "rgb(255, 99, 132)", borderColor: "rgb(255, 99, 132)",)lit";
        htmlData << "data: [";
        for(auto i = 0; i < itsTimestamps.size(); ++i) {
            htmlData << itsMemoryFootprints[i];
            if(i < itsTimestamps.size() - 1) {
                htmlData << ",";
            }
        }
        htmlData << "]}]}, ";
        htmlData << htmlPart2;
    }

    void reservedMemoryEvent(uint64_t memoryFootprint) {
        memoryConsumption += (memoryFootprint / 1000000000.0);
        itsMemoryFootprints.push_back(memoryConsumption);
        auto current = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(current - itsStart);
        itsTimestamps.push_back(duration.count());
    }

    void freedMemoryEvent(uint64_t memoryFootprint) {
        memoryConsumption -= (memoryFootprint / 1000000000.0);
        itsMemoryFootprints.push_back(memoryConsumption);
        auto current = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(current - itsStart);
        itsTimestamps.push_back(duration.count());
    }

private:
    EventWatcher() {
        itsStart = high_resolution_clock::now();
        memoryConsumption = 0;
    }

public:
    EventWatcher(EventWatcher const&)  = delete;
    void operator=(EventWatcher const&) = delete;

private:
    uint64_t step = 0;
    double memoryConsumption;
    std::vector<double> itsMemoryFootprints;
    std::vector<uint64_t> itsTimestamps;
    std::vector<std::string> itsEventLabels;
    time_point<steady_clock> itsStart;
};

#endif //OCTREECUDA_EVENTWATCHER_H
